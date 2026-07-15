"use client";

import { useEffect, useId, useRef } from "react";

const displayHeading = "font-display";

export default function DatasetDialog({
  open,
  onClose,
  images,
  isLoading,
  error,
  selectedName,
  selectingUrl,
  onSelect,
  onRetry,
}) {
  const titleId = useId();
  const closeButtonRef = useRef(null);
  const panelRef = useRef(null);

  useEffect(() => {
    if (!open) {
      return undefined;
    }

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeButtonRef.current?.focus();

    function handleKeyDown(event) {
      if (event.key === "Escape") {
        onClose();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [open, onClose]);

  if (!open) {
    return null;
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-end justify-center sm:items-center sm:p-6"
      role="presentation"
    >
      <button
        type="button"
        aria-label="Close dataset dialog"
        className="absolute inset-0 bg-[rgba(25,40,31,0.48)] backdrop-blur-[2px]"
        onClick={onClose}
      />

      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="relative z-10 flex max-h-[92dvh] w-full flex-col overflow-hidden rounded-t-[28px] border border-line bg-[rgba(255,250,242,0.98)] shadow-panel sm:max-h-[min(820px,88vh)] sm:max-w-3xl sm:rounded-[28px] md:max-w-4xl"
      >
        <div className="flex shrink-0 items-start justify-between gap-3 border-b border-[rgba(25,40,31,0.08)] px-4 pb-4 pt-[max(1rem,env(safe-area-inset-top))] sm:px-6 sm:pt-5">
          <div className="min-w-0">
            <p className="m-0 text-[0.8rem] font-bold uppercase tracking-[0.18em] text-primary">
              Online dataset
            </p>
            <h2
              id={titleId}
              className={`${displayHeading} mt-1 text-[clamp(1.35rem,4vw,1.85rem)] leading-tight`}
            >
              Choose a sample leaf image
            </h2>
            <p className="mt-1 text-sm text-muted">
              Tap an image to use it for diagnosis. You can still upload your own photo instead.
            </p>
          </div>
          <button
            ref={closeButtonRef}
            type="button"
            onClick={onClose}
            className="inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-full border border-[rgba(25,40,31,0.1)] bg-white/80 text-lg font-bold text-ink transition hover:bg-white"
            aria-label="Close"
          >
            ×
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-4 py-4 sm:px-6">
          {isLoading ? (
            <div className="flex min-h-48 flex-col items-center justify-center gap-3 text-muted">
              <div className="h-9 w-9 animate-spin rounded-full border-2 border-primary/20 border-t-primary" />
              <p className="m-0">Loading dataset images...</p>
            </div>
          ) : null}

          {error ? (
            <div className="flex min-h-48 flex-col items-center justify-center gap-4 text-center">
              <p className="m-0 max-w-md rounded-2xl bg-[rgba(179,62,42,0.1)] px-4 py-3.5 text-[#8d2f20]">
                {error}
              </p>
              <button
                type="button"
                onClick={onRetry}
                className="inline-flex min-h-11 items-center justify-center rounded-full border border-[rgba(25,40,31,0.1)] bg-white/80 px-5 py-2.5 text-ink transition hover:-translate-y-0.5"
              >
                Try again
              </button>
            </div>
          ) : null}

          {!isLoading && !error ? (
            images.length ? (
              <div className="grid grid-cols-2 gap-2.5 xs:grid-cols-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
                {images.map((image) => {
                  const isSelected = selectedName === image.name;
                  const isSelecting = selectingUrl === image.url;

                  return (
                    <button
                      key={image.url}
                      type="button"
                      disabled={Boolean(selectingUrl)}
                      className={`group relative grid cursor-pointer gap-1.5 rounded-2xl border bg-white/85 p-1.5 text-left transition active:scale-[0.98] disabled:cursor-wait sm:p-2 ${
                        isSelected
                          ? "border-primary shadow-[0_0_0_2px_rgba(31,111,67,0.18)]"
                          : "border-[rgba(25,40,31,0.12)] hover:border-primary/40 hover:shadow-sm"
                      }`}
                      onClick={() => onSelect(image)}
                    >
                      <span className="relative aspect-square overflow-hidden rounded-xl bg-[rgba(31,111,67,0.06)]">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={image.url}
                          alt={image.name}
                          loading="lazy"
                          className="block h-full w-full object-cover transition duration-200 group-hover:scale-[1.03]"
                        />
                        {isSelecting ? (
                          <span className="absolute inset-0 flex items-center justify-center bg-[rgba(255,250,242,0.72)]">
                            <span className="h-7 w-7 animate-spin rounded-full border-2 border-primary/20 border-t-primary" />
                          </span>
                        ) : null}
                        {isSelected && !isSelecting ? (
                          <span className="absolute right-1.5 top-1.5 rounded-full bg-primary px-2 py-0.5 text-[0.65rem] font-bold text-white">
                            Selected
                          </span>
                        ) : null}
                      </span>
                      <span className="truncate px-0.5 text-[0.7rem] leading-tight text-muted sm:text-xs">
                        {image.name}
                      </span>
                    </button>
                  );
                })}
              </div>
            ) : (
              <div className="flex min-h-48 items-center justify-center text-muted">
                <p className="m-0">No dataset images found in the directory.</p>
              </div>
            )
          ) : null}
        </div>

        <div className="flex shrink-0 items-center justify-between gap-3 border-t border-[rgba(25,40,31,0.08)] px-4 py-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] sm:px-6 sm:py-4">
          <p className="m-0 text-sm text-muted">
            {images.length ? `${images.length} images available` : "Browse sample leaves"}
          </p>
          <button
            type="button"
            onClick={onClose}
            className="inline-flex min-h-11 min-w-24 items-center justify-center rounded-full border border-[rgba(25,40,31,0.1)] bg-white/80 px-5 py-2.5 font-medium text-ink transition hover:-translate-y-0.5"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
