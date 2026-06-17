#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";
import { AqemWebStack } from "../lib/aqem-web-stack";

const app = new cdk.App();

// Account/region come from the ambient AWS credentials (CDK_DEFAULT_*).
// CloudFront's managed prefix-list lookup needs a concrete region, so we pin
// one (defaults to us-east-1, matching the rest of the project).
const env = {
  account: process.env.CDK_DEFAULT_ACCOUNT,
  region: process.env.CDK_DEFAULT_REGION ?? process.env.AWS_REGION ?? "us-east-1",
};

new AqemWebStack(app, "AqemWebStack", {
  env,
  description:
    "Agentic QEM Lab web UI + API — CloudFront -> ALB -> ECS Fargate (aqem-web)",
});
