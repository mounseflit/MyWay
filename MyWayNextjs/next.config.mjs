/**
 * Run `build` or `dev` with `SKIP_ENV_VALIDATION` to skip env validation. This is especially useful
 * for Docker builds.
 */
await import("./src/env.mjs");

/** @type {import("next").NextConfig} */
const config = {
    images: {
        domains: ['d3nvy39jvu7woe.cloudfront.net'], // Add the hostname here
      },
};

export default config;
