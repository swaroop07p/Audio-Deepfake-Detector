import { useState, useCallback, useContext, useRef, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { AuthContext } from "../context/AuthContext";
import { ScanContext } from "../context/ScanContext";
import { useNavigate } from "react-router-dom";
import ScannerOverlay from "../components/ScannerOverlay";
import ResultsView from "../components/ResultsView";
import Hero from "../components/Hero";
// 1. Import API
import api from '../api';
import {
  FaCloudUploadAlt,
  FaMicrophoneAlt,
  FaCheckCircle,
  FaLock,
} from "react-icons/fa";

const Background = () => (
  <>
    <div className="aurora-bg"></div>
    <div className="wave-container">
      <div className="wave"></div>
      <div className="wave"></div>
    </div>
  </>
);

const Home = () => {
  const { user } = useContext(AuthContext); 
  const { scanResult, setScanResult } = useContext(ScanContext);
  const navigate = useNavigate();
  const resultsRef = useRef(null);

  const [file, setFile] = useState(null);
  const [isScanning, setIsScanning] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (scanResult && resultsRef.current) {
        setTimeout(() => {
            resultsRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    }
  }, [scanResult]);

  const onDrop = useCallback((acceptedFiles) => {
    setFile(acceptedFiles[0]);
    setError("");
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "audio/*": [] },
    maxFiles: 1,
  });

  const handleAnalyze = async () => {
    // Check if logged in (Token is handled automatically by api.js now)
    if (!localStorage.getItem("token") && !user) {
      alert("You must be logged in to scan files.");
      navigate("/");
      return;
    }

    if (!file) return;

    setIsScanning(true);
    setError("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      // 2. FIXED: Use 'api.post' instead of 'axios.post'
      const response = await api.post("/api/detect", formData, {
          headers: {
            "Content-Type": "multipart/form-data",
            // No Authorization header needed here, api.js does it!
          },
      });

      setTimeout(() => {
        setScanResult(response.data);
        setIsScanning(false);
        setFile(null);
      }, 2000);
    } catch (err) {
      console.error("Upload detailed error:", err);
      setIsScanning(false);
      if (err.response && err.response.status === 401) {
        setError("Session expired. Please logout and login again.");
      } else if (err.code === "ERR_NETWORK") {
        setError("Cannot connect to server. Is the backend running?");
      } else {
        setError("Analysis failed. Please try a different file.");
      }
    }
  };

  const handleAnimationComplete = () => {
    if (scanResult || error) {
      setIsScanning(false);
    }
  };

  if (scanResult && !isScanning) {
    return (
      <div className="min-h-[100dvh] pt-24 pb-10 relative overflow-x-hidden" ref={resultsRef}>
        <Background />
        <div className="relative z-10">
            <ResultsView result={scanResult} />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-[100dvh] pt-24 pb-10 px-4 flex flex-col items-center justify-center relative overflow-hidden">
      
      <ScannerOverlay
        isScanning={isScanning}
        onComplete={handleAnimationComplete}
      />
      
      <Background />

      <div className="absolute top-0 left-0 w-full h-full z-[2] bg-gradient-to-b from-transparent via-black/10 to-black/40 pointer-events-none"></div>

      <div className="relative z-10 w-full flex flex-col items-center">
          <Hero />

          <div className="w-full max-w-2xl glass-panel p-6 md:p-10 rounded-3xl relative group transition-all duration-300 hover:shadow-[0_0_40px_rgba(0,243,255,0.2)] mt-8">
            <div
              {...getRootProps()}
              className={`
                border-2 border-dashed rounded-2xl h-56 md:h-64 flex flex-col items-center justify-center cursor-pointer transition-all
                ${isDragActive ? "border-neon-green bg-green-900/10" : "border-gray-600 hover:border-neon-blue hover:bg-white/5"}
            `}
            >
              <input {...getInputProps()} />
              {file ? (
                <div className="text-center p-4">
                  <FaMicrophoneAlt className="text-4xl md:text-5xl text-neon-green mb-4 mx-auto animate-bounce" />
                  <p className="text-lg md:text-xl font-bold text-white break-all">
                    {file.name}
                  </p>
                  <p className="text-sm text-gray-400 mt-2">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              ) : (
                <div className="text-center space-y-4 p-4">
                  <FaCloudUploadAlt className="text-5xl md:text-6xl text-gray-500 mx-auto group-hover:text-neon-blue transition-colors" />
                  <p className="text-base md:text-lg text-gray-300">
                    Drag & Drop Audio File
                  </p>
                  <p className="text-xs md:text-sm text-gray-500">
                    or click to browse (WAV, MP3, FLAC)
                  </p>
                </div>
              )}
            </div>

            {file && (
              <button
                onClick={handleAnalyze}
                className="w-full mt-6 py-4 rounded-xl bg-gradient-to-r from-neon-blue to-purple-600 font-bold text-lg tracking-wider hover:opacity-90 transition shadow-lg shadow-blue-500/30 flex justify-center items-center gap-2"
              >
                <FaLock /> INITIATE FORENSIC SCAN
              </button>
            )}

            {error && (
              <p className="text-red-500 mt-4 text-center bg-red-900/20 p-2 rounded">
                {error}
              </p>
            )}
          </div>

          <div className="mt-12 flex flex-wrap justify-center gap-4 md:gap-8 text-gray-500 text-xs md:text-sm font-mono">
            <span className="flex items-center gap-2">
              <FaCheckCircle /> Metadata Scan
            </span>
            <span className="flex items-center gap-2">
              <FaCheckCircle /> Spectral Analysis
            </span>
            <span className="flex items-center gap-2">
              <FaCheckCircle /> Biometric Match
            </span>
          </div>
      </div>
    </div>
  );
};

export default Home;