import type { Metadata } from "next";
import StemFlowStudio from "./stem-flow-studio";

export const metadata: Metadata = {
  title: "StemFlow — AI Stem Studio",
  description: "Upload audio, separate six stems, shape the mix, and export.",
};

export default function Home() {
  return <StemFlowStudio />;
}
