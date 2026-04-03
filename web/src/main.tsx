import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "./index.css";
import App from "./App.tsx";

// Loaded only on demand, so the game page never pulls in the benchmark
const Page = new URLSearchParams(location.search).has("bench") ? (await import("./Bench.tsx")).default : App;

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Page />
  </StrictMode>,
);
