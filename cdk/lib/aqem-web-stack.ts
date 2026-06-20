import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as ecs from "aws-cdk-lib/aws-ecs";
import * as elbv2 from "aws-cdk-lib/aws-elasticloadbalancingv2";
import * as cloudfront from "aws-cdk-lib/aws-cloudfront";
import * as origins from "aws-cdk-lib/aws-cloudfront-origins";
import * as logs from "aws-cdk-lib/aws-logs";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as iam from "aws-cdk-lib/aws-iam";
import * as path from "path";
import { Construct } from "constructs";

/**
 * AqemWebStack — public, cloud-native deployment of the Agentic QEM Lab web UI.
 *
 *   Browser --HTTPS--> CloudFront --HTTP(+secret header)--> ALB --> Fargate task
 *
 * A single Fargate task runs `aqem-web` (FastAPI + Uvicorn on :8000), which
 * serves both the built React SPA (ui/dist) and the /api/* endpoints. The
 * adaptive QEM loop runs in-process on a local Braket noise simulator; Bedrock
 * is only called when the user toggles "Use VLM" on, which the task role allows.
 *
 * CloudFront gives a free public HTTPS URL (https://<id>.cloudfront.net) and
 * injects a secret header the ALB checks, so the ALB cannot be reached directly.
 */
export class AqemWebStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // Secret header so the ALB only serves requests that came through CloudFront.
    const cfSecretHeaderName = "X-CloudFront-Secret";
    const cfSecretHeaderValue = cdk.Fn.select(
      2,
      cdk.Fn.split("/", `${cdk.Aws.STACK_ID}`)
    );

    // ===========================
    // S3 artifact bucket (audit logs / figures, parity with the cloud loop)
    // ===========================
    const artifactBucket = new s3.Bucket(this, "ArtifactBucket", {
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      enforceSSL: true,
      // Artifacts are disposable, so `cdk destroy` cleans up fully.
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    // ===========================
    // VPC
    // ===========================
    // Dedicated VPC with public subnets only and NO NAT gateway: the Fargate
    // task runs in a public subnet with a public IP for egress (ECR pull,
    // Bedrock), while its security group allows inbound only from the ALB — so
    // it is never directly reachable. Skipping the NAT gateway avoids consuming
    // an Elastic IP (the account is at its EIP quota) and is cheaper.
    const vpc = new ec2.Vpc(this, "Vpc", {
      maxAzs: 2,
      natGateways: 0,
      subnetConfiguration: [
        { cidrMask: 24, name: "Public", subnetType: ec2.SubnetType.PUBLIC },
      ],
    });

    // ===========================
    // ALB security group — ingress only from the CloudFront prefix list
    // ===========================
    const albSg = new ec2.SecurityGroup(this, "AlbSecurityGroup", {
      vpc,
      description: "ALB SG - HTTP from CloudFront managed prefix list only",
      allowAllOutbound: true,
    });

    const cfPrefixList = ec2.PrefixList.fromLookup(this, "CloudFrontPrefixList", {
      prefixListName: "com.amazonaws.global.cloudfront.origin-facing",
    });

    albSg.addIngressRule(
      ec2.Peer.prefixList(cfPrefixList.prefixListId),
      ec2.Port.tcp(80),
      "Allow HTTP from CloudFront managed prefix list"
    );

    // ===========================
    // ECS cluster + task definition
    // ===========================
    const cluster = new ecs.Cluster(this, "Cluster", {
      vpc,
      containerInsightsV2: ecs.ContainerInsights.ENABLED,
    });

    const taskDefinition = new ecs.FargateTaskDefinition(this, "TaskDef", {
      // The adaptive loop + kaleido PNG render are CPU-bound; 1 vCPU / 2 GB.
      cpu: 1024,
      memoryLimitMiB: 2048,
      runtimePlatform: {
        cpuArchitecture: ecs.CpuArchitecture.X86_64,
        operatingSystemFamily: ecs.OperatingSystemFamily.LINUX,
      },
    });

    // Task role: read/write artifacts + call the Bedrock VLM (when toggled on).
    artifactBucket.grantReadWrite(taskDefinition.taskRole);
    taskDefinition.taskRole.addToPrincipalPolicy(
      new iam.PolicyStatement({
        actions: [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream",
        ],
        // Claude models + the inference profiles the VLM resolves to.
        resources: [
          `arn:aws:bedrock:*::foundation-model/anthropic.claude-*`,
          `arn:aws:bedrock:*:${this.account}:inference-profile/*anthropic.claude-*`,
        ],
      })
    );

    const logGroup = new logs.LogGroup(this, "AppLogGroup", {
      logGroupName: "/ecs/aqem-web",
      retention: logs.RetentionDays.TWO_WEEKS,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // Build the UI+API image from Dockerfile.web at the repo root. The build
    // context is the repo root (one level up from cdk/).
    const repoRoot = path.join(__dirname, "..", "..");
    const container = taskDefinition.addContainer("AqemWeb", {
      image: ecs.ContainerImage.fromAsset(repoRoot, {
        file: "Dockerfile.web",
        platform: cdk.aws_ecr_assets.Platform.LINUX_AMD64,
      }),
      logging: ecs.LogDrivers.awsLogs({ logGroup, streamPrefix: "aqem-web" }),
      environment: {
        AWS_REGION: this.region,
        AQEM_DEVICE: "qd_readout_2",
        AQEM_ARTIFACTS: `s3://${artifactBucket.bucketName}/aqem`,
        // Record/replay cache for runs — S3-backed so it survives task restarts
        // and is shared across the autoscaled tasks (grantReadWrite below already
        // covers Get/Put/Delete/List).
        AQEM_CACHE: `s3://${artifactBucket.bucketName}/aqem-cache`,
        // VLM model used when a user toggles "Use VLM" on (Claude Opus 4.8).
        AQEM_VLM_MODEL_ID: "us.anthropic.claude-opus-4-8",
        PYTHONUNBUFFERED: "1",
      },
    });

    container.addPortMappings({
      containerPort: 8000,
      protocol: ecs.Protocol.TCP,
    });

    // ===========================
    // ECS service security group
    // ===========================
    const serviceSg = new ec2.SecurityGroup(this, "ServiceSecurityGroup", {
      vpc,
      description: "ECS service SG - traffic from the ALB only",
      allowAllOutbound: true,
    });
    serviceSg.addIngressRule(albSg, ec2.Port.tcp(8000), "Allow traffic from ALB");

    // ===========================
    // Application Load Balancer
    // ===========================
    const alb = new elbv2.ApplicationLoadBalancer(this, "ALB", {
      vpc,
      internetFacing: true,
      securityGroup: albSg,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      // Adaptive runs stream over SSE for a while — keep the connection open.
      idleTimeout: cdk.Duration.seconds(300),
    });
    alb.setAttribute("routing.http.drop_invalid_header_fields.enabled", "true");

    const targetGroup = new elbv2.ApplicationTargetGroup(this, "TargetGroup", {
      vpc,
      port: 8000,
      protocol: elbv2.ApplicationProtocol.HTTP,
      targetType: elbv2.TargetType.IP,
      healthCheck: {
        path: "/api/health",
        interval: cdk.Duration.seconds(15),
        timeout: cdk.Duration.seconds(5),
        healthyThresholdCount: 2,
        unhealthyThresholdCount: 3,
        healthyHttpCodes: "200",
      },
      deregistrationDelay: cdk.Duration.seconds(30),
    });

    const listener = alb.addListener("HttpListener", {
      port: 80,
      protocol: elbv2.ApplicationProtocol.HTTP,
      // Don't auto-open 0.0.0.0/0 — the SG already allows only the CloudFront
      // managed prefix list, so direct internet access never reaches the ALB.
      open: false,
      defaultAction: elbv2.ListenerAction.fixedResponse(403, {
        contentType: "text/plain",
        messageBody: "Forbidden - direct access not allowed",
      }),
    });

    listener.addAction("ForwardWithSecret", {
      priority: 1,
      conditions: [
        elbv2.ListenerCondition.httpHeader(cfSecretHeaderName, [
          cfSecretHeaderValue,
        ]),
      ],
      action: elbv2.ListenerAction.forward([targetGroup]),
    });

    // ===========================
    // Fargate service + autoscaling
    // ===========================
    const service = new ecs.FargateService(this, "Service", {
      cluster,
      taskDefinition,
      desiredCount: 1,
      minHealthyPercent: 100,
      maxHealthyPercent: 200,
      securityGroups: [serviceSg],
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      // Public IP gives the task egress via the IGW (no NAT GW / EIP needed).
      // Inbound is still locked to the ALB SG, so the task isn't reachable.
      assignPublicIp: true,
      circuitBreaker: { rollback: true },
      // A run pins one task's CPU; give health checks room before first probe.
      healthCheckGracePeriod: cdk.Duration.seconds(120),
    });
    service.attachToApplicationTargetGroup(targetGroup);

    const scaling = service.autoScaleTaskCount({
      minCapacity: 1,
      maxCapacity: 3,
    });
    scaling.scaleOnCpuUtilization("CpuScaling", {
      targetUtilizationPercent: 70,
      scaleInCooldown: cdk.Duration.seconds(120),
      scaleOutCooldown: cdk.Duration.seconds(60),
    });

    // ===========================
    // CloudFront distribution (the public HTTPS URL)
    // ===========================
    const albOrigin = new origins.HttpOrigin(alb.loadBalancerDnsName, {
      protocolPolicy: cloudfront.OriginProtocolPolicy.HTTP_ONLY,
      customHeaders: { [cfSecretHeaderName]: cfSecretHeaderValue },
      // SSE responses can take a while to start/finish; raise origin timeouts.
      readTimeout: cdk.Duration.seconds(60),
      keepaliveTimeout: cdk.Duration.seconds(60),
    });

    const responseHeadersPolicy = new cloudfront.ResponseHeadersPolicy(
      this,
      "SecurityHeaders",
      {
        responseHeadersPolicyName: `${this.stackName}-Security`,
        securityHeadersBehavior: {
          contentTypeOptions: { override: true },
          frameOptions: {
            frameOption: cloudfront.HeadersFrameOption.SAMEORIGIN,
            override: true,
          },
          referrerPolicy: {
            referrerPolicy:
              cloudfront.HeadersReferrerPolicy.STRICT_ORIGIN_WHEN_CROSS_ORIGIN,
            override: true,
          },
          strictTransportSecurity: {
            accessControlMaxAge: cdk.Duration.days(365),
            includeSubdomains: true,
            override: true,
          },
        },
      }
    );

    const distribution = new cloudfront.Distribution(this, "Distribution", {
      defaultBehavior: {
        origin: albOrigin,
        viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
        allowedMethods: cloudfront.AllowedMethods.ALLOW_ALL,
        cachePolicy: cloudfront.CachePolicy.CACHING_DISABLED,
        originRequestPolicy: cloudfront.OriginRequestPolicy.ALL_VIEWER,
        responseHeadersPolicy,
      },
      additionalBehaviors: {
        // POST /api/run + SSE must pass through uncached and unbuffered.
        "/api/*": {
          origin: albOrigin,
          viewerProtocolPolicy:
            cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
          allowedMethods: cloudfront.AllowedMethods.ALLOW_ALL,
          cachePolicy: cloudfront.CachePolicy.CACHING_DISABLED,
          originRequestPolicy: cloudfront.OriginRequestPolicy.ALL_VIEWER,
          responseHeadersPolicy,
        },
      },
      priceClass: cloudfront.PriceClass.PRICE_CLASS_100,
      httpVersion: cloudfront.HttpVersion.HTTP2_AND_3,
      comment: "Agentic QEM Lab web UI",
    });

    // ===========================
    // Outputs
    // ===========================
    new cdk.CfnOutput(this, "CloudFrontURL", {
      value: `https://${distribution.distributionDomainName}`,
      description: "Public URL — open this to use the AQEM Lab",
    });
    new cdk.CfnOutput(this, "ALBDnsName", {
      value: alb.loadBalancerDnsName,
      description: "ALB DNS (direct access returns 403)",
    });
    new cdk.CfnOutput(this, "ArtifactBucketName", {
      value: artifactBucket.bucketName,
      description: "S3 bucket for run artifacts",
    });
  }
}
