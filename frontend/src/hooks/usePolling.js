import { useEffect, useRef } from "react";

export function usePolling(callback, delay, immediate = true) {
  const callbackRef = useRef(callback);

  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);

  useEffect(() => {
    let cancelled = false;

    async function tick() {
      if (!cancelled) {
        await callbackRef.current();
      }
    }

    if (immediate) {
      tick();
    }

    const timer = window.setInterval(tick, delay);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [delay, immediate]);
}
