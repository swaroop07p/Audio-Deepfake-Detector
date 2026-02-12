import { useState, useCallback, useContext, useRef, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { AuthContext } from "../context/AuthContext";
import { ScanContext } from "../context/ScanContext";
import { useNavigate } from "react-router-dom";
import ScannerOverlay from "../components/ScannerOverlay";
import ResultsView from "../components/ResultsView";
import Hero from "../components/Hero";
import api from '../api'; // Central API
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

  // Scroll to results when they appear
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
    // Clear previous results when a new file is dropped
    setScanResult(null); 
  }, [setScanResult]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "audio/*": [] },
    maxFiles: 1,
  });

  const handleAnalyze = async () => {
    // 1. Check Login
    if (!localStorage.getItem("token") && !user) {
      alert("You must be logged in to scan files.");
      navigate("/");
      return;
    }

    if (!file) return;

    // 2. Start Animation
    setIsScanning(true);
    setError("");
    setScanResult(null); // Ensure result is clear before starting

    const formData = new FormData();
    formData.append("file", file);

    try {
      // 3. Send Request
      console.log("Sending request to backend..."); // Debug Log
      const response = await api.post("/api/detect", formData, {
          headers: { "Content-Type": "multipart/form-data" },
      });

      console.log("Response received:", response.data); // Debug Log

      // 4. Stop Animation & Show Result
      // We wait 2 seconds just for the 'cool' animation effect, then show data
      setTimeout(() => {
        setScanResult(response.data);
        setIsScanning(false);
        setFile(null);
      }, 2000);

    } catch (err) {
      console.error("Analysis Failed:", err);
      setIsScanning(false); // STOP ANIMATION ON ERROR
      
      if (err.response) {
        if (err.response.status === 401) {
            setError("Session expired. Please login again.");
            localStorage.clear();
            navigate('/');
        } else if (err.response.status === 413) {
            setError("File is too large.");
        } else {
            setError("Server Error: " + (err.response.data.detail || "Unknown"));
        }
      } else if (err.code === "ERR_NETWORK") {
        setError("Cannot connect to server. Is the backend running?");
      } else {
        setError("Analysis failed. Please try a different file.");
      }
    }
  };

  // This handles the "ScannerOverlay" telling us it's done visually
  const handleAnimationComplete = () => {
    // Only stop if we actually have a result or error. 
    // If backend is still thinking, keep scanning!
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
      
      {/* Pass scanResult so Overlay knows when to finish */}
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
                disabled={isScanning} // Disable button while scanning
                className={`w-full mt-6 py-4 rounded-xl font-bold text-lg tracking-wider flex justify-center items-center gap-2 transition shadow-lg
                    ${isScanning ? 'bg-gray-700 cursor-not-allowed text-gray-400' : 'bg-gradient-to-r from-neon-blue to-purple-600 hover:opacity-90 shadow-blue-500/30'}
                `}
              >
                {isScanning ? (
                    <>Scanning...</>
                ) : (
                    <><FaLock /> INITIATE FORENSIC SCAN</>
                )}
              </button>
            )}

            {error && (
              <p className="text-red-500 mt-4 text-center bg-red-900/20 p-2 rounded border border-red-500/50">
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