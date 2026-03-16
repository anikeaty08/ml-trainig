import { startTransition, useEffect, useMemo, useState } from "react";
import { Route, Routes } from "react-router-dom";

import Header from "./components/Header";
import Sidebar from "./components/Sidebar";
import Dashboard from "./pages/Dashboard";
import ModelSelection from "./pages/ModelSelection";
import Processing from "./pages/Processing";
import Results from "./pages/Results";
import Upload from "./pages/Upload";
import { usePolling } from "./hooks/usePolling";
import { api } from "./services/api";

const defaultSettings = {
  provider: "ollama",
  auth_mode: "local",
  base_url: "http://127.0.0.1:11434",
  primary_model_ref: "ollama/llama3.2",
  image_model_ref: "ollama/llava:7b",
  model: "",
  fallback_models: "",
  fallback_model_refs: "",
  model_allowlist: "",
  model_catalog: [
    {
      alias: "local-fast",
      ref: "ollama/llama3.2",
      capability: "chat",
      description: "Fast local default"
    },
    {
      alias: "local-vision",
      ref: "ollama/llava:7b",
      capability: "image",
      description: "Local image-capable default"
    }
  ],
  auth_profiles: [],
  auth_order: {},
  browser_session_hint: "",
  api_key: ""
};

export default function App() {
  const [jobs, setJobs] = useState([]);
  const [availableModels, setAvailableModels] = useState({ builtin_models: [], candidate_families: [] });
  const [providerCatalog, setProviderCatalog] = useState([]);
  const [settings, setSettings] = useState(defaultSettings);
  const [configLoaded, setConfigLoaded] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function loadMeta() {
      const [models, providers, storedConfig] = await Promise.all([
        api.getAvailableModels(),
        api.getProviders(),
        api.getStoredConfig()
      ]);
      if (!cancelled) {
        setAvailableModels(models);
        setProviderCatalog(providers.providers || []);
        setSettings({ ...defaultSettings, ...storedConfig });
        setConfigLoaded(true);
      }
    }

    loadMeta();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!configLoaded) {
      return;
    }
    const timer = window.setTimeout(() => {
      api.saveStoredConfig(settings).catch(() => {});
    }, 250);
    return () => window.clearTimeout(timer);
  }, [configLoaded, settings]);

  usePolling(async () => {
    const data = await api.listJobs();
    startTransition(() => setJobs(data));
  }, 3000);

  const activeModelRef = useMemo(() => {
    if (settings.primary_model_ref) {
      return settings.primary_model_ref;
    }
    if (!settings.model) {
      return settings.provider || "unset";
    }
    return settings.model.includes("/") ? settings.model : `${settings.provider}/${settings.model}`;
  }, [settings]);

  return (
    <div className="app-shell">
      <Sidebar />
      <main className="main-panel">
        <Header activeModelRef={activeModelRef} jobCount={jobs.length} />
        <Routes>
          <Route
            element={
              <Dashboard
                availableModels={availableModels}
                jobs={jobs}
                onSettingsChange={setSettings}
                providerCatalog={providerCatalog}
                settings={settings}
              />
            }
            path="/"
          />
          <Route
            element={
              <ModelSelection
                onSettingsChange={setSettings}
                providerCatalog={providerCatalog}
                settings={settings}
              />
            }
            path="/models"
          />
          <Route element={<Upload settings={settings} />} path="/upload" />
          <Route element={<Processing />} path="/processing/:jobId" />
          <Route element={<Results />} path="/results/:jobId" />
        </Routes>
      </main>
    </div>
  );
}
