import { useEffect, useState } from "react";
import { checkBackend } from "./services/api";

function App() {
  const [status, setStatus] = useState("Checking...");

  useEffect(() => {
    checkBackend()
      .then((data) => setStatus(data.status))
      .catch(() => setStatus("offline"));
  }, []);

  return (
    <main style={{ padding: "40px" }}>
      <h1>SASTRA AI</h1>
      <p>Intelligent Student Assistant</p>

      <h3>AI Backend: {status}</h3>
    </main>
  );
}

export default App;