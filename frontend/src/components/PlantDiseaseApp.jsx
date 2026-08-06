"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import DatasetDialog from "@/components/DatasetDialog";
import {
  apiBaseUrl,
  backendKeepAliveIntervalMs,
  demoImagesApiUrl,
  getApiErrorMessage,
  readJsonSafely,
} from "@/lib/api";

const preventionTips = [
  "Inspect leaves weekly and isolate infected plants early.",
  "Avoid watering foliage late in the day to reduce fungal spread.",
  "Rotate crops and clean tools between plant beds.",
  "Use balanced fertilizer to support natural plant resistance.",
];

const deploymentCards = [
  {
    title: "Frontend on Vercel",
    text: "Next.js + Tailwind ships with fast global delivery and simple environment variables.",
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

const glassPanel =
  "rounded-[28px] border border-line bg-surface p-6 shadow-panel backdrop-blur-[18px] max-sm:rounded-[22px] max-sm:p-5";

const primaryButton =
  "inline-flex min-h-12 items-center justify-center rounded-full border-0 bg-gradient-to-br from-primary to-[#2f8a55] px-5 py-3.5 text-white shadow-[0_18px_42px_rgba(31,111,67,0.24)] transition duration-150 ease-out hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-70";

const secondaryButton =
  "inline-flex min-h-12 items-center justify-center rounded-full border border-[rgba(25,40,31,0.1)] bg-white/70 px-5 py-3.5 text-ink transition duration-150 ease-out hover:-translate-y-0.5";

const ghostButton =
  "inline-flex min-h-12 items-center justify-center rounded-full border border-[rgba(25,40,31,0.1)] bg-white/70 px-5 py-3.5 text-ink transition duration-150 ease-out hover:-translate-y-0.5";

const eyebrow = "m-0 text-[0.9rem] font-bold uppercase tracking-[0.18em] text-primary";

const displayHeading = "font-display";

export default function PlantDiseaseApp() {
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
  const [selectingDatasetUrl, setSelectingDatasetUrl] = useState("");
  const previewUrlRef = useRef("");
  const shouldScrollToResultRef = useRef(false);

  function updateSelectedFile(file) {
    setSelectedFile(file);
    setResult(null);
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
    }
    const nextPreviewUrl = file ? URL.createObjectURL(file) : "";
    previewUrlRef.current = nextPreviewUrl;
    setPreviewUrl(nextPreviewUrl);
  }

  useEffect(() => {
    return () => {
      if (previewUrlRef.current) {
        URL.revokeObjectURL(previewUrlRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!shouldScrollToResultRef.current) {
      return;
    }
    if (!isLoading && !result) {
      return;
    }

    shouldScrollToResultRef.current = false;
    document.getElementById("prediction-result")?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }, [isLoading, result]);

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
      } catch {
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
        await fetch(`${apiBaseUrl}/api/warmup`, { method: "POST" });
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

  const cropsPreview = useMemo(() => catalog.supportedCrops.slice(0, 8), [catalog.supportedCrops]);

  async function loadDatasetImages() {
    setDatasetError("");
    setIsDatasetLoading(true);

    try {
      const response = await fetch(demoImagesApiUrl);
      const data = await readJsonSafely(response);

      if (!response.ok || !Array.isArray(data)) {
        throw new Error("Failed to load dataset directory.");
      }

      const images = data
        .map((item) => ({
          name: item.name || item.filename,
          url: item.url.startsWith("http") ? item.url : `${apiBaseUrl}${item.url}`,
        }))
        .sort((imageA, imageB) => imageA.name.localeCompare(imageB.name));

      setDatasetImages(images);
    } catch (loadError) {
      setDatasetError(getApiErrorMessage(loadError, "Failed to load dataset directory."));
    } finally {
      setIsDatasetLoading(false);
    }
  }

  const closeDatasetDialog = useCallback(() => {
    setIsDatasetOpen(false);
    setSelectingDatasetUrl("");
  }, []);

  async function openDatasetDialog() {
    setIsDatasetOpen(true);
    if (datasetImages.length === 0 && !isDatasetLoading) {
      await loadDatasetImages();
    }
  }

  async function handleDatasetImageSelect(image) {
    setError("");
    setSelectingDatasetUrl(image.url);

    try {
      const response = await fetch(image.url);
      if (!response.ok) {
        throw new Error("Failed to download selected dataset image.");
      }
      const imageBlob = await response.blob();
      const fileType =
        imageBlob.type || (image.name.toLowerCase().endsWith(".png") ? "image/png" : "image/jpeg");
      const imageFile = new File([imageBlob], image.name, { type: fileType });
      updateSelectedFile(imageFile);
      setApiNotice("");
      closeDatasetDialog();
    } catch (selectionError) {
      setError(getApiErrorMessage(selectionError, "Failed to use selected dataset image."));
      setSelectingDatasetUrl("");
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
    shouldScrollToResultRef.current = true;

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
    <div className="relative overflow-hidden">
      <div className="pointer-events-none absolute -left-40 -top-32 h-[34rem] w-[34rem] rounded-full bg-[rgba(122,185,110,0.45)] opacity-45 blur-[32px]" />
      <div className="pointer-events-none absolute -right-48 top-48 h-[34rem] w-[34rem] rounded-full bg-[rgba(215,122,43,0.28)] opacity-45 blur-[32px]" />

      <header className="relative z-10 mx-auto w-[min(1180px,calc(100%-2rem))] py-6 pb-10 max-sm:w-[min(100%-1rem,100%)]">
        <nav className="mb-8 flex items-start justify-between gap-4 max-[980px]:flex-col">
          <div>
            <p className={eyebrow}>Plant Disease Detection</p>
            <h1
              className={`${displayHeading} mt-1.5 max-w-[12ch] text-[clamp(2.6rem,5vw,5rem)] leading-[0.96] max-sm:text-[2.55rem]`}
            >
              AI-assisted crop diagnosis with a production-ready deployment split.
            </h1>
          </div>
          <a className={`${ghostButton} w-full sm:w-auto`} href="#analyzer">
            Run Diagnosis
          </a>
        </nav>

        <div className="grid grid-cols-[minmax(0,1.25fr)_minmax(320px,0.9fr)] gap-5 max-[980px]:grid-cols-1">
          <section className={`${glassPanel} animate-fade-up`}>
            <p className={eyebrow}>Created by Nishant Makwana</p>
            <p className="mt-3 text-muted">
              Upload a plant leaf image, call the Render API, and get disease insights powered by a
              PyTorch model hosted through Hugging Face.
            </p>
            <div className="mt-4 grid gap-3 sm:flex sm:flex-wrap sm:items-center">
              <a className={`${primaryButton} w-full sm:w-auto`} href="#analyzer">
                Start Prediction
              </a>
              <a className={`${secondaryButton} w-full sm:w-auto`} href="#creator">
                About Creator
              </a>
            </div>
            <div className="mt-6 grid grid-cols-3 gap-5 max-[980px]:grid-cols-1">
              <article className="rounded-[18px] bg-white/[0.66] p-4">
                <strong className="mb-0.5 block text-[1.8rem]">{catalog.totalClasses || 39}</strong>
                <span>Model classes</span>
              </article>
              <article className="rounded-[18px] bg-white/[0.66] p-4">
                <strong className="mb-0.5 block text-[1.8rem]">
                  {catalog.supportedCrops.length || 14}
                </strong>
                <span>Supported crops</span>
              </article>
              <article className="rounded-[18px] bg-white/[0.66] p-4">
                <strong className="mb-0.5 block text-[1.8rem]">3</strong>
                <span>Free services</span>
              </article>
            </div>
          </section>

          <aside className={`${glassPanel} animate-fade-up-delay`}>
            <div className="animate-pulse-soft w-fit rounded-full bg-[rgba(17,71,42,0.08)] px-3.5 py-2 font-bold text-primary-dark">
              Optimized for desktop and mobile screens
            </div>
            <div className="mt-5 grid grid-cols-2 gap-5 max-sm:grid-cols-1">
              {cropsPreview.map((crop) => (
                <span key={crop} className="rounded-[18px] bg-white/70 px-4 py-3">
                  {crop}
                </span>
              ))}
            </div>
            <div className="mt-5 grid justify-items-center gap-3 rounded-[22px] bg-gradient-to-b from-white/88 to-[rgba(246,239,225,0.82)] p-4 font-bold">
              <span>User</span>
              <div className="h-[26px] w-0.5 bg-gradient-to-b from-accent to-primary" />
              <span>Vercel Frontend</span>
              <div className="h-[26px] w-0.5 bg-gradient-to-b from-accent to-primary" />
              <span>Render API</span>
              <div className="h-[26px] w-0.5 bg-gradient-to-b from-accent to-primary" />
              <span>Hugging Face Model</span>
            </div>
          </aside>
        </div>
      </header>

      <main>
        <section className="relative z-10 mx-auto w-[min(1180px,calc(100%-2rem))] animate-fade-up pb-6 max-sm:w-[min(100%-1rem,100%)]">
          <div className="mb-4">
            <p className={eyebrow}>Deployment structure</p>
            <h2 className={`${displayHeading} mt-1.5 text-[clamp(1.7rem,2.6vw,2.4rem)]`}>
              Free hosting split with clear responsibility boundaries.
            </h2>
          </div>
          <div className="grid grid-cols-3 gap-5 max-[980px]:grid-cols-1">
            {deploymentCards.map((card) => (
              <article key={card.title} className={glassPanel}>
                <h3 className={`${displayHeading} mb-1.5 mt-0 text-[1.3rem]`}>{card.title}</h3>
                <p className="text-muted">{card.text}</p>
              </article>
            ))}
          </div>
        </section>

        <section
          className="relative z-10 mx-auto grid w-[min(1180px,calc(100%-2rem))] grid-cols-1 gap-5 pb-6 max-sm:w-[min(100%-1rem,100%)]"
          id="analyzer"
        >
          <div className={`${glassPanel} animate-fade-up`}>
            <div className="mb-4">
              <p className={eyebrow}>Analyzer</p>
              <h2 className={`${displayHeading} mt-1.5 text-[clamp(1.7rem,2.6vw,2.4rem)]`}>
                {selectedFile ? "Image ready — run prediction" : "Start by choosing a leaf image"}
              </h2>
              <p className="mt-2 text-sm text-muted sm:text-base">
                {selectedFile
                  ? "Review the preview below, then tap Predict to get your diagnosis."
                  : "Upload your own photo or pick a sample from the online dataset."}
              </p>
            </div>

            <form onSubmit={handleSubmit} className="grid gap-4">
              {!selectedFile ? (
                <>
                  <label
                    className="grid min-h-[7.5rem] cursor-pointer place-content-center gap-1.5 rounded-3xl border-[1.5px] border-dashed border-[rgba(31,111,67,0.28)] bg-white/70 p-5 text-center sm:min-h-0 sm:place-content-stretch sm:text-left"
                    htmlFor="leaf-upload"
                  >
                    <input
                      id="leaf-upload"
                      type="file"
                      accept="image/png,image/jpeg,image/jpg"
                      className="hidden"
                      onChange={(event) => updateSelectedFile(event.target.files?.[0] || null)}
                    />
                    <span className="font-bold">Drop a leaf image here or tap to browse</span>
                    <span className="text-sm text-muted sm:text-base">
                      PNG, JPG, or JPEG up to 10 MB
                    </span>
                  </label>

                  <div className="relative flex items-center gap-3">
                    <div className="h-px flex-1 bg-[rgba(25,40,31,0.1)]" />
                    <span className="text-xs font-bold uppercase tracking-[0.16em] text-muted">
                      or
                    </span>
                    <div className="h-px flex-1 bg-[rgba(25,40,31,0.1)]" />
                  </div>

                  <button
                    className={`${secondaryButton} w-full`}
                    type="button"
                    onClick={openDatasetDialog}
                  >
                    Use online dataset images
                  </button>
                </>
              ) : (
                <div className="grid gap-4 rounded-[24px] border border-[rgba(31,111,67,0.16)] bg-white/65 p-3 sm:p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <p className="m-0 text-sm font-bold text-primary">Selected image</p>
                      <p className="m-0 mt-1 truncate text-sm text-muted sm:text-base">
                        {selectedFile.name}
                      </p>
                    </div>
                    <button
                      type="button"
                      className="shrink-0 rounded-full border border-[rgba(25,40,31,0.1)] bg-white/80 px-3 py-2 text-sm font-medium text-ink transition hover:-translate-y-0.5"
                      onClick={() => {
                        updateSelectedFile(null);
                        setError("");
                      }}
                    >
                      Change
                    </button>
                  </div>

                  {previewUrl ? (
                    <div className="aspect-[4/3] overflow-hidden rounded-[20px] bg-white/70 sm:aspect-[16/10] sm:max-h-[360px]">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={previewUrl}
                        alt="Leaf preview"
                        className="block h-full w-full object-cover"
                      />
                    </div>
                  ) : null}

                  <button
                    className={`${primaryButton} w-full`}
                    type="submit"
                    disabled={isLoading}
                  >
                    {isLoading ? "Analyzing image..." : "Predict Plant Disease"}
                  </button>

                  {!isLoading ? (
                    <div className="grid gap-2 sm:flex sm:flex-wrap">
                      <label
                        className={`${secondaryButton} w-full cursor-pointer sm:w-auto`}
                        htmlFor="leaf-upload-change"
                      >
                        <input
                          id="leaf-upload-change"
                          type="file"
                          accept="image/png,image/jpeg,image/jpg"
                          className="hidden"
                          onChange={(event) =>
                            updateSelectedFile(event.target.files?.[0] || null)
                          }
                        />
                        Upload another
                      </label>
                      <button
                        className={`${secondaryButton} w-full sm:w-auto`}
                        type="button"
                        onClick={openDatasetDialog}
                      >
                        Browse dataset
                      </button>
                    </div>
                  ) : null}
                </div>
              )}

              {error ? (
                <p className="m-0 rounded-2xl bg-[rgba(179,62,42,0.1)] px-4 py-3.5 text-[#8d2f20]">
                  {error}
                </p>
              ) : null}
              {!error && apiNotice && !result ? (
                <p className="m-0 rounded-2xl bg-white/50 px-4 py-3.5 text-sm sm:text-base">
                  {apiNotice}
                </p>
              ) : null}
            </form>
          </div>

          {isLoading || result ? (
            <div
              id="prediction-result"
              className={`${glassPanel} animate-fade-up scroll-mt-4`}
            >
              <div className="mb-4">
                <p className={eyebrow}>Prediction result</p>
                <h2 className={`${displayHeading} mt-1.5 text-[clamp(1.7rem,2.6vw,2.4rem)]`}>
                  {isLoading
                    ? "Analyzing your leaf image..."
                    : result?.displayName || "Diagnosis complete"}
                </h2>
              </div>

              {isLoading ? (
                <div className="flex min-h-40 flex-col items-center justify-center gap-3 text-muted">
                  <div className="h-9 w-9 animate-spin rounded-full border-2 border-primary/20 border-t-primary" />
                  <p className="m-0 text-center text-sm sm:text-base">
                    Running inference on the backend. This can take a moment on a cold start.
                  </p>
                </div>
              ) : null}

              {result && !isLoading ? (
                <div className="grid gap-4">
                  <div className="flex flex-wrap items-center justify-between gap-3.5 rounded-2xl bg-white/70 px-4 py-3">
                    <span className="text-muted">Confidence</span>
                    <strong className="text-xl text-primary">{result.confidence}%</strong>
                  </div>
                  <p className="m-0 text-muted">{result.description}</p>
                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                    <article className="rounded-2xl bg-white/60 p-4">
                      <h3 className={`${displayHeading} mb-1.5 mt-0 text-[1.2rem]`}>
                        {result.isHealthy ? "Plant care guidance" : "Prevention steps"}
                      </h3>
                      <p className="m-0 text-sm text-muted sm:text-base">{result.possibleSteps}</p>
                    </article>
                    <article className="rounded-2xl bg-white/60 p-4">
                      <h3 className={`${displayHeading} mb-1.5 mt-0 text-[1.2rem]`}>
                        {result.isHealthy ? "Recommended fertilizer" : "Recommended supplement"}
                      </h3>
                      <p className="m-0 text-sm text-muted sm:text-base">
                        {result.supplement?.name || "No supplement linked for this class."}
                      </p>
                      {result.supplement?.buyLink ? (
                        <a
                          className={`${secondaryButton} mt-3 w-full sm:w-fit`}
                          href={result.supplement.buyLink}
                          target="_blank"
                          rel="noreferrer"
                        >
                          View Product
                        </a>
                      ) : null}
                    </article>
                  </div>
                  {result.referenceImage ? (
                    <div className="aspect-[4/3] overflow-hidden rounded-3xl bg-white/70 sm:aspect-auto sm:min-h-[220px]">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={result.referenceImage}
                        alt={result.displayName}
                        className="block h-full w-full object-cover"
                      />
                    </div>
                  ) : null}
                </div>
              ) : null}
            </div>
          ) : null}
        </section>

        <section className="relative z-10 mx-auto w-[min(1180px,calc(100%-2rem))] animate-fade-up pb-6 max-sm:w-[min(100%-1rem,100%)]">
          <div className="grid grid-cols-2 gap-5 max-[980px]:grid-cols-1">
            <article className={glassPanel}>
              <div className="mb-4">
                <p className={eyebrow}>Healthy ops</p>
                <h2 className={`${displayHeading} mt-1.5 text-[clamp(1.7rem,2.6vw,2.4rem)]`}>
                  What changed in the codebase
                </h2>
              </div>
              <ul className="m-0 grid list-disc gap-4 pl-4 text-muted">
                <li>Frontend moved into a Next.js + Tailwind app ready for Vercel.</li>
                <li>Backend now exposes JSON APIs for Render deployment.</li>
                <li>Model loading supports Hugging Face Hub instead of committing large binaries.</li>
                <li>Project branding and creator details are updated across the app.</li>
              </ul>
            </article>

            <article className={glassPanel}>
              <div className="mb-4">
                <p className={eyebrow}>Prevention basics</p>
                <h2 className={`${displayHeading} mt-1.5 text-[clamp(1.7rem,2.6vw,2.4rem)]`}>
                  Field practices worth keeping
                </h2>
              </div>
              <ul className="m-0 grid list-disc gap-4 pl-4 text-muted">
                {preventionTips.map((tip) => (
                  <li key={tip}>{tip}</li>
                ))}
              </ul>
            </article>
          </div>
        </section>

        <section
          className="relative z-10 mx-auto w-[min(1180px,calc(100%-2rem))] animate-fade-up pb-6 max-sm:w-[min(100%-1rem,100%)]"
          id="creator"
        >
          <div className={glassPanel}>
            <div className="mb-4">
              <p className={eyebrow}>Creator</p>
              <h2 className={`${displayHeading} mt-1.5 text-[clamp(1.7rem,2.6vw,2.4rem)]`}>
                Project created by Nishant Makwana
              </h2>
            </div>
            <p className="text-muted">
              This refresh keeps the original plant diagnosis idea, but restructures it for modern free
              deployment and a stronger portfolio presentation.
            </p>
            <div className="mt-4 grid grid-cols-[repeat(3,max-content)] gap-5 max-[980px]:grid-cols-1">
              {creatorLinks.map((link) => (
                <a
                  key={link.label}
                  className={`${secondaryButton} w-fit`}
                  href={link.href}
                  target="_blank"
                  rel="noreferrer"
                >
                  {link.label}
                </a>
              ))}
            </div>
          </div>
        </section>
      </main>

      <DatasetDialog
        open={isDatasetOpen}
        onClose={closeDatasetDialog}
        images={datasetImages}
        isLoading={isDatasetLoading}
        error={datasetError}
        selectedName={selectedFile?.name || ""}
        selectingUrl={selectingDatasetUrl}
        onSelect={handleDatasetImageSelect}
        onRetry={loadDatasetImages}
      />
    </div>
  );
}
