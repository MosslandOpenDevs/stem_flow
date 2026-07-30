import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "StemFlow — AI Stem Studio",
  description: "Upload audio, separate every stem, and create your own mix.",
  icons: { icon: "/favicon.svg" },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
