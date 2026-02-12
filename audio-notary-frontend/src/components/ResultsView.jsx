import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { FaFilePdf, FaExclamationTriangle, FaUserSecret, FaRedo } from 'react-icons/fa';
import api from '../api';
import { useContext } from 'react';
import { AuthContext } from '../context/AuthContext';
import { ScanContext } from '../context/ScanContext';
import { toast } from 'react-toastify';

const ResultsView = ({ result }) => {
  const { user } = useContext(AuthContext);
  const { resetScan } = useContext(ScanContext);

  const isFake = result.verdict === "AI/Synthetic";
  const fakeScore = result.confidence_score;
  const humanScore = 100 - fakeScore;

  // Decide what to show (The winning score)
  const displayScore = isFake ? fakeScore : humanScore;
  const displayLabel = isFake ? "AI Probability" : "Human Probability";

  const pieData = [
    { name: 'Real Human', value: humanScore },
    { name: 'AI Synthetic', value: fakeScore },
  ];
  const COLORS = ['#00ff9d', '#ff0055'];

  // --- CUSTOM TOOLTIP FIX ---
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-black/90 border border-white/20 p-4 rounded-xl shadow-[0_0_20px_rgba(0,0,0,0.8)] backdrop-blur-md">
          <p className="text-white font-bold mb-1">{payload[0].name}</p>
          <p className="text-gray-300">
            Confidence: <span style={{ color: payload[0].payload.fill, fontWeight: 'bold' }}>
              {payload[0].value.toFixed(1)}%
            </span>
          </p>
        </div>
      );
    }
    return null;
  };

  const handleDownload = async () => {
    if (!user || user.user_type === 'guest') {
        toast.error("Download restricted. Please login as a User.");
        return;
    }
    if (!result._id) {
        toast.error("Report not saved. Please check History.");
        return;
    }
    try {
        const response = await api.get(`/api/report/${result._id}/download`, {
            responseType: 'blob',
        });
        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', `Forensic_Report_${result.filename}.pdf`);
        document.body.appendChild(link);
        link.click();
        toast.success("Report downloaded successfully!");
    } catch (error) {
        toast.error("Download failed.");
    }
  };

  return (
    <div className="w-full max-w-6xl mx-auto px-4 md:px-6 space-y-8 animate-fade-in-up pb-20">
      
      {/* 1. Verdict Banner */}
      <div className={`p-6 md:p-10 rounded-3xl text-center border-2 shadow-[0_0_30px_rgba(0,0,0,0.5)] ${isFake ? 'bg-red-900/40 border-neon-red shadow-neon-red/20' : 'bg-green-900/40 border-neon-green shadow-neon-green/20'}`}>
        <h1 className="text-3xl md:text-5xl font-black uppercase tracking-widest mb-3 text-white drop-shadow-md flex justify-center items-center gap-4 flex-wrap">
          {isFake ? <FaExclamationTriangle className="text-neon-red"/> : <FaUserSecret className="text-neon-green"/>}
          <span>{isFake ? "SYNTHETIC AUDIO DETECTED" : "VERIFIED HUMAN VOICE"}</span>
        </h1>
        <p className="text-gray-300 font-mono text-lg md:text-xl">
            {displayLabel}: <span className={`font-bold ${isFake ? 'text-neon-red' : 'text-neon-green'}`}>{displayScore.toFixed(2)}%</span>
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-8">
        
        {/* 2. Biometric Graph */}
        <div className="glass-panel p-6 rounded-2xl flex flex-col items-center">
            <h3 className="text-xl font-bold mb-6 text-neon-blue flex items-center gap-2 uppercase tracking-wide">
                <FaUserSecret/> Biometric Analysis
            </h3>
            <div className="w-full h-64 relative">
                <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                        <Pie
                            data={pieData}
                            cx="50%"
                            cy="50%"
                            innerRadius={60}
                            outerRadius={90}
                            paddingAngle={5}
                            dataKey="value"
                            stroke="none"
                        >
                            {pieData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                        </Pie>
                        <Tooltip content={<CustomTooltip />} />
                    </PieChart>
                </ResponsiveContainer>
                
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-center pointer-events-none">
                    <span className={`block text-3xl font-bold ${isFake ? 'text-neon-red' : 'text-neon-green'}`}>
                        {displayScore.toFixed(0)}%
                    </span>
                    <span className="text-xs text-gray-400 whitespace-nowrap">{isFake ? 'AI PROBABILITY' : 'HUMAN PROBABILITY'}</span>
                </div>
            </div>
            <div className="flex justify-center gap-6 mt-6 text-sm font-bold">
                <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-neon-green"></span> Real Human</div>
                <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-neon-red"></span> AI Generated</div>
            </div>
        </div>

        {/* 3. Forensic Insights */}
        <div className="glass-panel p-6 rounded-2xl">
            <h3 className="text-xl font-bold mb-6 text-neon-blue flex items-center gap-2 uppercase tracking-wide">
                <FaExclamationTriangle/> Forensic Insights
            </h3>
            <ul className="space-y-4">
                {result.reasons && result.reasons.length > 0 ? (
                    result.reasons.map((reason, idx) => (
                        <li key={idx} className="flex items-start gap-3 bg-white/5 p-4 rounded-xl border-l-4 border-neon-blue hover:bg-white/10 transition">
                            <span className="mt-1 text-neon-blue text-lg">➤</span>
                            <p className="text-sm md:text-base text-gray-200 leading-relaxed">{reason}</p>
                        </li>
                    ))
                ) : (
                    <li className="flex items-start gap-3 bg-white/5 p-4 rounded-xl border-l-4 border-green-500">
                        <span className="mt-1 text-green-500 text-lg">✔</span>
                        <p className="text-sm md:text-base text-gray-200 leading-relaxed">
                            No anomalies detected. Audio passed all forensic checks.
                        </p>
                    </li>
                )}
            </ul>
        </div>
      </div>

      {/* 4. Buttons */}
      <div className="flex flex-col md:flex-row justify-center gap-4 mt-8">
        <button 
            onClick={resetScan}
            className="w-full md:w-auto px-8 py-4 rounded-full bg-gray-800 hover:bg-gray-700 transition font-bold border border-gray-600 flex items-center justify-center gap-2"
        >
            <FaRedo /> Analyze New File
        </button>
        
        {user?.user_type !== 'guest' ? (
            <button 
                onClick={handleDownload}
                className="w-full md:w-auto px-8 py-4 rounded-full bg-gradient-to-r from-neon-blue to-purple-600 hover:scale-105 transition font-bold flex items-center justify-center gap-2 shadow-lg shadow-blue-500/30"
            >
                <FaFilePdf /> Download Notarized Report
            </button>
        ) : (
            <div className="group relative w-full md:w-auto">
                <button disabled className="w-full md:w-auto px-8 py-4 rounded-full bg-gray-800 text-gray-500 cursor-not-allowed font-bold flex items-center justify-center gap-2 border border-gray-700">
                    <FaFilePdf /> Download (Login Required)
                </button>
            </div>
        )}
      </div>
    </div>
  );
};

export default ResultsView;