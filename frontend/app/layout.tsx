import type { Metadata } from "next";
import { CopilotKit } from "@copilotkit/react-core";
import "@copilotkit/react-ui/styles.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "GeoVision — Geospatial Image Analysis",
  description:
    "Conversational geospatial analysis powered by multi-agent YOLO inference",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <CopilotKit runtimeUrl="/api/copilotkit" agent="geoVisionAgent">
          {children}
        </CopilotKit>
      </body>
    </html>
  );
}
