import type { NextConfig } from "next";

// A fully static site: `next build` writes plain HTML/CSS/JS to out/. No server runs
// anywhere, so there is nothing to attack or to pay for. Data is fetched in the browser.
const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true, // /konum → /konum/index.html, works on any static host
  images: { unoptimized: true }, // image optimization needs a server
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
