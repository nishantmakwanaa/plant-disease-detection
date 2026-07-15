export const apiBaseUrl = (
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:5000"
).replace(/\/$/, "");

export const backendKeepAliveIntervalMs = 30 * 60 * 1000;

export const datasetTreeApiUrl =
  "https://huggingface.co/api/spaces/nishantmakwanaa/plant-disease-detection-app/tree/main/test_images";

export const datasetImageBaseUrl =
  "https://huggingface.co/spaces/nishantmakwanaa/plant-disease-detection-app/resolve/main/test_images";

export async function readJsonSafely(response) {
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
