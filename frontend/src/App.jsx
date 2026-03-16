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

const SETTINGS_KEY = "ml-agent-provider-settings";

const defaultSettings = {
  provider: "ollama",
  auth_mode: "local",
  base_url: "http://127.0.0.1:11434",
  model: "",
  fallback_models: "",
  api_key: ""
};

function loadSettings() {
  try {
    return { ...defaultSettings, ...JSON.parse(window.localStorage.getItem(SETTINGS_KEY) || "{}") };
  } catch (error) {
    return defaultSettings;
  }
}

export default function App() {
  const [jobs, setJobs] = useState([]);
  const [availableModels, setAvailableModels] = useState({ builtin_models: [], candidate_families: [] });
  const [providerCatalog, setProviderCatalog] = useState([]);
  const [settings, setSettings] = useState(loadSettings);

  useEffect(() => {
    window.localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  }, [settings]);

  useEffect(() => {
    let cancelled = false;

    async function loadMeta() {
      const [models, providers] = await Promise.all([api.getAvailableModels(), api.getProviders()]);
      if (!cancelled) {
        setAvailableModels(models);
        setProviderCatalog(providers.providers || []);
      }
    }

    loadMeta();
    return () => {
      cancelled = true;
    };
  }, []);

  usePolling(async () => {
    const data = await api.listJobs();
    startTransition(() => setJobs(data));
  }, 3000);

  const activeModelRef = useMemo(() => {
    if (!settings.model) {
      return settings.provider || "unset";
    }
    return `${settings.provider}/${settings.model}`;
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
