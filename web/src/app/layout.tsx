import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";

import { SiteFooter, SiteHeader } from "@/components/Nav";
import { HtmlLang } from "@/lib/i18n";

import "./globals.css";

// latin-ext has the Turkish letters ğ, ş, ı, İ. The fonts are downloaded at build time and
// served from this site: visitors' browsers never contact Google.
const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin", "latin-ext"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin", "latin-ext"] });

export const metadata: Metadata = {
  title: "SpotCast · AI-powered water sports forecasts",
  description:
    "Is this spot right for my level, at this hour? SpotCast evaluates wind and wave forecasts for windsurf, kitesurf, SUP and sea kayak along the Turkish coast with a machine learning model and strict safety rules.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col font-sans">
        <HtmlLang />
        <SiteHeader />
        <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-10">{children}</main>
        <SiteFooter />
      </body>
    </html>
  );
}
