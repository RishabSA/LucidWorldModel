export interface Input {
	// One action per tick, 0 (noop) when there is none
	next: () => number;
	clear: () => void;
	stop: () => void;
}

const queueDepth = 3;

// Keys map KeyboardEvent.code to an action. Queue mode plays every press in order; hold mode plays the most recently pressed key still down
export function listen(
	keys: Record<string, number>,
	mode: "queue" | "hold",
): Input {
	let pressed: string[] = [];

	const onKeyDown = (event: KeyboardEvent) => {
		// Leave browser shortcuts such as Cmd+W alone
		if (event.ctrlKey || event.metaKey || event.altKey) return;
		if (!Object.hasOwn(keys, event.code)) return;
		event.preventDefault();
		if (event.repeat) return;

		if (mode === "hold") {
			pressed = pressed.filter(code => code !== event.code);
			pressed.push(event.code);
		} else if (pressed.length < queueDepth) {
			pressed.push(event.code);
		}
	};
	const onKeyUp = (event: KeyboardEvent) => {
		if (mode === "hold") pressed = pressed.filter(code => code !== event.code);
	};
	const clear = () => {
		pressed = [];
	};

	window.addEventListener("keydown", onKeyDown);
	window.addEventListener("keyup", onKeyUp);
	// Keyups are lost while the window is unfocused, so held keys would stick
	window.addEventListener("blur", clear);

	return {
		next: () => {
			const code = mode === "queue" ? pressed.shift() : pressed.at(-1);
			return code === undefined ? 0 : keys[code];
		},
		clear,
		stop: () => {
			window.removeEventListener("keydown", onKeyDown);
			window.removeEventListener("keyup", onKeyUp);
			window.removeEventListener("blur", clear);
		},
	};
}
