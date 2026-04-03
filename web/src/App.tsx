function App() {
  return (
    <main className="space-y-4 p-8">
      <h1 className="text-3xl font-bold">LucidWorldModel</h1>
      <p>
        Games arrive in Stage 4. The browser latency benchmark is at{" "}
        <a
          href="/?bench"
          className="cursor-pointer text-blue-600 underline transition-colors hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300"
        >
          /?bench
        </a>
        .
      </p>
    </main>
  );
}

export default App;
