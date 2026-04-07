import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App.tsx";
import "./index.css";

const params = new URLSearchParams(location.search);
// Loaded only on demand, so the game page never pulls in the benchmark or the parity check
const Page = params.has("bench")
	? (await import("./Bench.tsx")).default
	: params.has("parity")
		? (await import("./Parity.tsx")).default
		: App;

createRoot(document.getElementById("root")!).render(
	<StrictMode>
		<Page />
	</StrictMode>,
);
