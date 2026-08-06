export const apiBaseUrl = (
  process.env.NEXT_PUBLIC_API_BASE_URL || "https://nishantmakwanaa-plant-disease-detection-app.hf.space"
).replace(/\/$/, "");

export const backendKeepAliveIntervalMs = 5 * 60 * 1000; // 5 minutes

export const datasetImagesApiUrl = `${apiBaseUrl}/api/test-dataset`;
export const demoImagesApiUrl = datasetImagesApiUrl;

export async function readJsonSafely(response) {
  if (!response || !response.ok) {
    return null;
  }
  const contentType = response.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) {
    return null;
  }

  try {
    return await response.json();
  } catch {
    return null;
  }
}

export function getApiErrorMessage(error, fallbackMessage) {
  if (error instanceof Error && error.message) {
    if (error.message === "Failed to fetch") {
      return "The backend is unreachable or still waking up. Wait a moment and try again.";
    }
    return error.message;
  }

  return fallbackMessage;
}

