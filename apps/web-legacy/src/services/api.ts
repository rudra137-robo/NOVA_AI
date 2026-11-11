const API_URL =
  import.meta.env.VITE_API_BASE_URL;

export async function checkBackend() {
  const response = await fetch(
    `${API_URL}/api/v1/health`
  );

  if (!response.ok) {
    throw new Error("Backend unavailable");
  }

  return response.json();
}