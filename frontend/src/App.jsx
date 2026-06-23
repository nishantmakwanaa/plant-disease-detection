import { useEffect, useMemo, useState } from "react";


const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL || "http://localhost:5000").replace(/\/$/, "");
const backendKeepAliveIntervalMs = 30 * 60 * 1000;
const datasetTreeApiUrl =
  "https://huggingface.co/api/spaces/nishantmakwanaa/plant-disease-detection-app/tree/main/test_images";
const datasetImageBaseUrl =
  "https://huggingface.co/spaces/nishantmakwanaa/plant-disease-detection-app/resolve/main/test_images";


async function readJsonSafely(response) {
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


function getApiErrorMessage(error, fallbackMessage) {
  if (error instanceof Error && error.message) {
    if (error.message === "Failed to fetch") {
      return "The backend is unreachable or still waking up. Wait a moment and try again.";
    }
    return error.message;
  }

  return fallbackMessage;
}

const preventionTips = [
  "Inspect leaves weekly and isolate infected plants early.",
  "Avoid watering foliage late in the day to reduce fungal spread.",
  "Rotate crops and clean tools between plant beds.",
  "Use balanced fertilizer to support natural plant resistance.",
];

const deploymentCards = [
  {
    title: "Frontend on Vercel",
    text: "React + Vite ships as a static build with fast global delivery and simple environment variables.",
  },
  {
    title: "Backend on Render",
    text: "FastAPI serves a dedicated inference API with health and prediction endpoints for the UI.",
  },
  {
    title: "Model on Hugging Face",
    text: "The backend downloads the PyTorch model from Hugging Face at startup, so the repo stays lightweight.",
  },
];

const creatorLinks = [
  { label: "GitHub", href: "https://github.com/nishantmakwanaa" },
  { label: "LinkedIn", href: "https://linkedin.com/in/nishantmakwanaa" },
];

function App() {
  const [catalog, setCatalog] = useState({ supportedCrops: [], totalClasses: 39 });
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [apiNotice, setApiNotice] = useState("Preparing the backend for the first prediction.");
  const [isLoading, setIsLoading] = useState(false);
  const [isDatasetOpen, setIsDatasetOpen] = useState(false);
  const [datasetImages, setDatasetImages] = useState([]);
  const [isDatasetLoading, setIsDatasetLoading] = useState(false);
  const [datasetError, setDatasetError] = useState("");

  useEffect(() => {
    let ignore = false;

    async function loadCatalog() {
      try {
        const response = await fetch(`${apiBaseUrl}/api/catalog`);
        const data = await readJsonSafely(response);
        if (!response.ok) {
          throw new Error(data?.error || "Catalog request failed.");
        }
        if (!ignore) {
          setCatalog(data);
        }
      } catch (_catalogError) {
        if (!ignore) {
          setCatalog({ supportedCrops: [], totalClasses: 39 });
        }
      }
    }

    async function warmupBackend() {
      try {
        const response = await fetch(`${apiBaseUrl}/api/warmup`, { method: "POST" });
        const data = await readJsonSafely(response);

        if (ignore) {
          return;
        }

        if (response.ok && response.status !== 202) {
          setApiNotice("");
          return;
        }

        setApiNotice(
          data?.message || "Backend cold start detected. The first prediction can take a little longer."
        );
      } catch {
        if (!ignore) {
          setApiNotice("Backend cold start detected. If prediction fails, wait a moment and retry.");
        }
      }
    }

    async function keepBackendAlive() {
      try {
        await fetch(`${apiBaseUrl}/api/health`, { cache: "no-store" });
      } catch {
        // Ignore keepalive failures; user-triggered requests will surface errors.
      }
    }

    loadCatalog();
    warmupBackend();
    const keepAliveTimer = setInterval(keepBackendAlive, backendKeepAliveIntervalMs);

    return () => {
      ignore = true;
      clearInterval(keepAliveTimer);
    };
  }, []);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl("");
      return undefined;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);

    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [selectedFile]);

  const cropsPreview = useMemo(() => catalog.supportedCrops.slice(0, 8), [catalog.supportedCrops]);

  async function loadDatasetImages() {
    setDatasetError("");
    setIsDatasetLoading(true);

    try {
      const response = await fetch(datasetTreeApiUrl);
      const data = await readJsonSafely(response);

      if (!response.ok || !Array.isArray(data)) {
        throw new Error("Failed to load dataset directory.");
      }

      const images = data
        .filter((item) => item?.type === "file" && /\.(png|jpe?g|webp)$/i.test(item?.path || ""))
        .map((item) => {
          const relativePath = item.path.replace(/^test_images\//, "");
          const encodedPath = relativePath.split("/").map(encodeURIComponent).join("/");
          return {
            name: relativePath.split("/").pop() || relativePath,
            url: `${datasetImageBaseUrl}/${encodedPath}`,
          };
        })
        .sort((imageA, imageB) => imageA.name.localeCompare(imageB.name));

      setDatasetImages(images);
    } catch (loadError) {
      setDatasetError(getApiErrorMessage(loadError, "Failed to load dataset directory."));
    } finally {
      setIsDatasetLoading(false);
    }
  }

  async function handleDatasetToggle() {
    const nextOpenState = !isDatasetOpen;
    setIsDatasetOpen(nextOpenState);
    if (nextOpenState && datasetImages.length === 0 && !isDatasetLoading) {
      await loadDatasetImages();
    }
  }

  async function handleDatasetImageSelect(image) {
    setError("");

    try {
      const response = await fetch(image.url);
      if (!response.ok) {
        throw new Error("Failed to download selected dataset image.");
      }
      const imageBlob = await response.blob();
      const fileType = imageBlob.type || (image.name.toLowerCase().endsWith(".png") ? "image/png" : "image/jpeg");
      const imageFile = new File([imageBlob], image.name, { type: fileType });
      setSelectedFile(imageFile);
      setApiNotice("");
    } catch (selectionError) {
      setError(getApiErrorMessage(selectionError, "Failed to use selected dataset image."));
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();

    if (!selectedFile) {
      setError("Choose a leaf image before running the diagnosis.");
      return;
    }

    setError("");
    setIsLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(`${apiBaseUrl}/api/predict`, {
        method: "POST",
        body: formData,
      });
      const data = await readJsonSafely(response);
      if (!response.ok) {
        throw new Error(data?.error || "Prediction failed.");
      }
      setApiNotice("");
      setResult(data);
    } catch (submissionError) {
      setError(getApiErrorMessage(submissionError, "Prediction failed."));
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <div className="ambient ambient-left" />
      <div className="ambient ambient-right" />

      <header className="hero-section">
        <nav className="topbar">
          <div>
            <p className="eyebrow">Plant Disease Detection</p>
            <h1>AI-assisted crop diagnosis with a production-ready deployment split.</h1>
          </div>
          <a className="ghost-button" href="#analyzer">
            Run Diagnosis
          </a>
        </nav>

        <div className="hero-grid">
          <section className="hero-copy glass-panel reveal">
            <p className="eyebrow">Created by Nishant Makwana</p>
            <p className="hero-lead">
              Upload a plant leaf image, call the Render API, and get disease insights powered by a
              PyTorch model hosted through Hugging Face.
            </p>
            <div className="hero-actions">
              <a className="primary-button" href="#analyzer">
                Start Prediction
              </a>
              <a className="secondary-button" href="#creator">
                About Creator
              </a>
            </div>
            <div className="stats-grid">
              <article>
                <strong>{catalog.totalClasses || 39}</strong>
                <span>Model classes</span>
              </article>
              <article>
                <strong>{catalog.supportedCrops.length || 14}</strong>
                <span>Supported crops</span>
              </article>
              <article>
                <strong>3</strong>
                <span>Free services</span>
              </article>
            </div>
          </section>

          <aside className="hero-showcase glass-panel reveal delay-1">
            <div className="pulse-chip">Optimized for desktop and mobile screens</div>
            <div className="crop-chip-grid">
              {cropsPreview.map((crop) => (
                <span key={crop}>{crop}</span>
              ))}
            </div>
            <div className="diagram-card">
              <span>User</span>
              <div className="diagram-arrow" />
              <span>Vercel Frontend</span>
              <div className="diagram-arrow" />
              <span>Render API</span>
              <div className="diagram-arrow" />
              <span>Hugging Face Model</span>
            </div>
          </aside>
        </div>
      </header>

      <main>
        <section className="content-section reveal">
          <div className="section-heading">
            <p className="eyebrow">Deployment structure</p>
            <h2>Free hosting split with clear responsibility boundaries.</h2>
          </div>
          <div className="card-grid three-up">
            {deploymentCards.map((card) => (
              <article key={card.title} className="info-card glass-panel">
                <h3>{card.title}</h3>
                <p>{card.text}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="content-section analyzer-grid" id="analyzer">
          <div className="glass-panel uploader-panel reveal">
            <div className="section-heading compact">
              <p className="eyebrow">Analyzer</p>
              <h2>Upload a leaf image and call the deployed FastAPI backend.</h2>
            </div>

            <form onSubmit={handleSubmit} className="upload-form">
              <label className="upload-dropzone" htmlFor="leaf-upload">
                <input
                  id="leaf-upload"
                  type="file"
                  accept="image/png,image/jpeg,image/jpg"
                  onChange={(event) => setSelectedFile(event.target.files?.[0] || null)}
                />
                <span className="upload-title">Drop a leaf image here or click to browse</span>
                <span className="upload-subtitle">PNG, JPG, or JPEG up to 10 MB</span>
              </label>

              <div className="dataset-toggle-row">
                <button className="secondary-button" type="button" onClick={handleDatasetToggle}>
                  {isDatasetOpen ? "Hide online dataset images" : "Use online dataset images"}
                </button>
              </div>

              {isDatasetOpen ? (
                <div className="dataset-browser">
                  {isDatasetLoading ? <p className="file-name">Loading dataset images...</p> : null}
                  {datasetError ? <p className="status-message error">{datasetError}</p> : null}

                  {!isDatasetLoading && !datasetError ? (
                    datasetImages.length ? (
                      <div className="dataset-grid">
                        {datasetImages.map((image) => (
                          <button
                            key={image.url}
                            type="button"
                            className={`dataset-image-button${selectedFile?.name === image.name ? " active" : ""}`}
                            onClick={() => handleDatasetImageSelect(image)}
                          >
                            <span className="dataset-thumb">
                              <img src={image.url} alt={image.name} loading="lazy" />
                            </span>
                            <span className="dataset-name">{image.name}</span>
                          </button>
                        ))}
                      </div>
                    ) : (
                      <p className="file-name">No dataset images found in the directory.</p>
                    )
                  ) : null}
                </div>
              ) : null}

              {selectedFile ? <p className="file-name">Selected: {selectedFile.name}</p> : null}

              {previewUrl ? (
                <div className="preview-frame">
                  <img src={previewUrl} alt="Leaf preview" />
                </div>
              ) : null}

              <button className="primary-button full-width" type="submit" disabled={isLoading}>
                {isLoading ? "Analyzing image..." : "Predict Plant Disease"}
              </button>

              {error ? <p className="status-message error">{error}</p> : null}
              {!error && apiNotice ? <p className="status-message">{apiNotice}</p> : null}
            </form>
          </div>

          <div className="glass-panel result-panel reveal delay-1">
            <div className="section-heading compact">
              <p className="eyebrow">Prediction result</p>
              <h2>{result ? result.displayName : "Your diagnosis will appear here."}</h2>
            </div>

            {result ? (
              <div className="result-stack">
                <div className="confidence-row">
                  <span className="confidence-label">Confidence</span>
                  <strong>{result.confidence}%</strong>
                </div>
                <p>{result.description}</p>
                <div className="split-panel">
                  <article>
                    <h3>{result.isHealthy ? "Plant care guidance" : "Prevention steps"}</h3>
                    <p>{result.possibleSteps}</p>
                  </article>
                  <article>
                    <h3>{result.isHealthy ? "Recommended fertilizer" : "Recommended supplement"}</h3>
                    <p>{result.supplement?.name || "No supplement linked for this class."}</p>
                    {result.supplement?.buyLink ? (
                      <a className="secondary-button inline-button" href={result.supplement.buyLink} target="_blank" rel="noreferrer">
                        View Product
                      </a>
                    ) : null}
                  </article>
                </div>
                {result.referenceImage ? (
                  <div className="reference-frame">
                    <img src={result.referenceImage} alt={result.displayName} />
                  </div>
                ) : null}
              </div>
            ) : (
              <div className="empty-state">
                <p>
                  The API response includes the disease label, confidence score, prevention guidance,
                  and supplement metadata from the original project dataset.
                </p>
              </div>
            )}
          </div>
        </section>

        <section className="content-section reveal">
          <div className="card-grid two-up">
            <article className="glass-panel info-card">
              <div className="section-heading compact">
                <p className="eyebrow">Healthy ops</p>
                <h2>What changed in the codebase</h2>
              </div>
              <ul className="feature-list">
                <li>Frontend moved into a standalone React app ready for Vercel.</li>
                <li>Backend now exposes JSON APIs for Render deployment.</li>
                <li>Model loading supports Hugging Face Hub instead of committing large binaries.</li>
                <li>Project branding and creator details are updated across the app.</li>
              </ul>
            </article>

            <article className="glass-panel info-card">
              <div className="section-heading compact">
                <p className="eyebrow">Prevention basics</p>
                <h2>Field practices worth keeping</h2>
              </div>
              <ul className="feature-list">
                {preventionTips.map((tip) => (
                  <li key={tip}>{tip}</li>
                ))}
              </ul>
            </article>
          </div>
        </section>

        <section className="content-section reveal" id="creator">
          <div className="glass-panel creator-panel">
            <div className="section-heading compact">
              <p className="eyebrow">Creator</p>
              <h2>Project created by Nishant Makwana</h2>
            </div>
            <p>
              This refresh keeps the original plant diagnosis idea, but restructures it for modern free
              deployment and a stronger portfolio presentation.
            </p>
            <div className="link-row">
              {creatorLinks.map((link) => (
                <a key={link.label} className="secondary-button inline-button" href={link.href} target="_blank" rel="noreferrer">
                  {link.label}
                </a>
              ))}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
