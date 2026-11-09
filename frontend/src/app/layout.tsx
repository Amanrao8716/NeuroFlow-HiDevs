import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";
import "@xyflow/react/dist/style.css";
import { Providers } from "@/components/providers";
import { Sidebar } from "@/components/layout/Sidebar";
import { Navbar } from "@/components/layout/Navbar";
import { ChunkDrawer } from "@/components/common/ChunkDrawer";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: "NeuroFlow — Observable RAG Platform",
  description: "Production observability, evaluation streaming, and management for RAG pipelines.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-slate-950 text-slate-100 flex min-h-screen overflow-x-hidden`}
      >
        <Providers>
          <Sidebar />
          <div className="flex-1 flex flex-col min-w-0">
            <Navbar />
            <main className="flex-1 p-6 md:p-8 overflow-y-auto">{children}</main>
          </div>
          <ChunkDrawer />
        </Providers>
      </body>
    </html>
  );
}
