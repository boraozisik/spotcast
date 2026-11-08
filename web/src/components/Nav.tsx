"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { useT } from "@/lib/i18n";
import { GITHUB_URL } from "@/lib/site";

import { LanguageSwitch } from "./LanguageSwitch";

export function SiteHeader() {
  const { t } = useT();
  const pathname = usePathname();
  const links = [
    { href: "/", text: t.nav.home },
    { href: "/konum/", text: t.nav.place },
    { href: "/ornekler/", text: t.nav.assistant },
    { href: "/degerlendirme/", text: t.nav.evaluation },
    { href: "/nasil-calisir/", text: t.nav.how },
  ];

  return (
    <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/90 backdrop-blur">
      <div className="mx-auto flex max-w-6xl flex-col gap-3 px-4 py-3 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex items-center justify-between gap-3">
          <Link href="/" className="flex items-center gap-2 font-semibold text-slate-900">
            <span aria-hidden className="text-2xl">
              🏄
            </span>
            <span>
              SpotCast <span className="font-normal text-slate-500">· {t.brand}</span>
            </span>
          </Link>
          <div className="lg:hidden">
            <LanguageSwitch />
          </div>
        </div>
        <div className="flex items-center gap-3">
          <nav className="flex gap-1 overflow-x-auto text-sm">
            {links.map(({ href, text }) => {
              const active =
                href === "/" ? pathname === "/" : pathname.startsWith(href.replace(/\/$/, ""));
              return (
                <Link
                  key={href}
                  href={href}
                  className={`whitespace-nowrap rounded-full px-3 py-1.5 transition ${
                    active ? "bg-sky-900 text-white" : "text-slate-600 hover:bg-slate-100"
                  }`}
                >
                  {text}
                </Link>
              );
            })}
          </nav>
          <div className="hidden lg:block">
            <LanguageSwitch />
          </div>
        </div>
      </div>
    </header>
  );
}

export function SiteFooter() {
  const { t } = useT();
  return (
    <footer className="border-t border-slate-200 bg-white">
      <div className="mx-auto flex max-w-6xl flex-col gap-2 px-4 py-6 text-sm text-slate-500 sm:flex-row sm:justify-between">
        <div className="space-y-1">
          <p>{t.footer.disclaimer}</p>
          <p>
            {t.footer.data}:{" "}
            <a href="https://open-meteo.com/" className="underline">
              Open-Meteo.com
            </a>{" "}
            (
            <a href="https://creativecommons.org/licenses/by/4.0/" className="underline">
              CC BY 4.0
            </a>
            )
          </p>
        </div>
        <a href={GITHUB_URL} className="font-medium text-sky-800 hover:underline">
          {t.footer.source} →
        </a>
      </div>
    </footer>
  );
}
