import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";

import {
  fetchPredictions,
  type Prediction,
} from "@/lib/predictions";

import { SOCHeader } from "@/components/soc/Header";
import { ThreatStatus } from "@/components/soc/ThreatStatus";
import { DeviceTelemetry } from "@/components/soc/DeviceTelemetry";
import { ModelAnalysis } from "@/components/soc/ModelAnalysis";
import { AlertBanner } from "@/components/soc/AlertBanner";
import { Charts } from "@/components/soc/Charts";
import { HistoryTable } from "@/components/soc/HistoryTable";

import {
  PRIMARY_MODEL_ID,
  getModel,
  type ModelId,
} from "@/lib/models";


export const Route = createFileRoute("/")({

  head: () => ({

    meta: [

      {
        title:
          "Threat Sentinel · IoT Anomaly & Malware Detection Platform",
      },

      {
        name: "description",

        content:
          "Real-time Security Operations Center dashboard for IoT anomaly and malware detection powered by a 3-model ensemble consisting of CNN-LSTM, Random Forest, and Autoencoder.",
      },

    ],

  }),

  component: Dashboard,

});


function Dashboard() {

  // =====================================================
  // STATE
  // =====================================================

  const [
    predictions,
    setPredictions,
  ] = useState<Prediction[]>([]);


  const [
    isLive,
    setIsLive,
  ] = useState(false);


  const [
    lastUpdate,
    setLastUpdate,
  ] = useState<Date | null>(null);


  const [
    error,
    setError,
  ] = useState<string | null>(null);


  const [
    selectedModel,
    setSelectedModel,
  ] = useState<ModelId>(
    PRIMARY_MODEL_ID
  );


  // =====================================================
  // FETCH PREDICTIONS
  // =====================================================

  useEffect(() => {

    let cancelled = false;


    async function load() {

      try {

        const data =
          await fetchPredictions();


        if (cancelled) {
          return;
        }


        setPredictions(data);

        setIsLive(true);

        setLastUpdate(
          new Date()
        );

        setError(null);

      }

      catch (e) {

        if (cancelled) {
          return;
        }


        setIsLive(false);


        setError(
          e instanceof Error
            ? e.message
            : "Unable to connect to ML server"
        );

      }

    }


    // Initial load
    load();


    // Refresh every 5 seconds
    const intervalId =
      setInterval(
        load,
        5000
      );


    return () => {

      cancelled = true;

      clearInterval(
        intervalId
      );

    };

  }, []);


  // =====================================================
  // LATEST ML DETECTION
  // =====================================================

  const latest =
    predictions[0] ?? null;


  // =====================================================
  // DASHBOARD
  // =====================================================

  return (

    <div
      className="
        min-h-screen
        bg-background
        text-foreground
        relative
      "
    >

      {/* ================================================= */}
      {/* BACKGROUND GRID                                   */}
      {/* ================================================= */}

      <div
        className="
          absolute
          inset-0
          bg-grid
          opacity-30
          pointer-events-none
        "
      />


      {/* ================================================= */}
      {/* BACKGROUND GRADIENT                               */}
      {/* ================================================= */}

      <div
        className="
          absolute
          inset-x-0
          top-0
          h-64
          bg-gradient-to-b
          from-primary/5
          to-transparent
          pointer-events-none
        "
      />


      {/* ================================================= */}
      {/* MAIN CONTENT                                      */}
      {/* ================================================= */}

      <main
        className="
          relative
          max-w-[1600px]
          mx-auto
          px-4
          md:px-6
          py-6
          space-y-8
        "
      >

        {/* ================================================= */}
        {/* 1. THREAT SENTINEL HEADER                         */}
        {/* ================================================= */}

        <SOCHeader
          isLive={isLive}
          lastUpdate={lastUpdate}
        />


        {/* ================================================= */}
        {/* API CONNECTION ERROR                              */}
        {/* ================================================= */}

        {error && (

          <div
            className="
              rounded-lg
              border
              border-danger/50
              bg-danger/10
              px-5
              py-3
              text-sm
              text-danger
              font-mono
            "
          >

            ⚠ Connection issue: {error}

            <span className="ml-1">
              Retrying…
            </span>

          </div>

        )}


        {/* ================================================= */}
        {/* 2. CURRENT THREAT STATUS                          */}
        {/* ================================================= */}

        <ThreatStatus
          latest={latest}
        />


        {/* ================================================= */}
        {/* 3. REAL ESP32 TELEMETRY                           */}
        {/* ================================================= */}

        <DeviceTelemetry />


        {/* ================================================= */}
        {/* 4. ML MODEL ANALYSIS (3-MODEL ENSEMBLE)           */}
        {/* ================================================= */}

        <ModelAnalysis
          latest={latest}
        />


        {/* ================================================= */}
        {/* ACTIVE THREAT ALERT (when attack)                 */}
        {/* ================================================= */}

        <AlertBanner
          latest={latest}
        />


        {/* ================================================= */}
        {/* 5. THREAT / ACTIVITY CHARTS                        */}
        {/* ================================================= */}

        <Charts
          predictions={predictions}
        />


        {/* ================================================= */}
        {/* 6. DETECTION HISTORY                              */}
        {/* ================================================= */}

        <HistoryTable
          predictions={predictions}
          activeModelName={
            getModel(
              selectedModel
            ).name
          }
        />


        {/* ================================================= */}
        {/* FOOTER                                            */}
        {/* ================================================= */}

        <footer
          className="
            text-center
            text-xs
            text-muted-foreground
            pt-8
            pb-4
            font-mono
            border-t
            border-border/40
          "
        >

          Threat Sentinel · Real-Time IoT Anomaly & Malware Detection Platform

        </footer>

      </main>

    </div>

  );

}