import { useState, useContext, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { AuthContext } from '../context/AuthContext';
import { ScanContext } from '../context/ScanContext'; 
import api from '../api'; // Use Central API
import { toast } from 'react-toastify';

const Login = () => {
  const [isRegister, setIsRegister] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [username, setUsername] = useState("");
  const { login, user } = useContext(AuthContext);
  const { resetScan } = useContext(ScanContext);
  const navigate = useNavigate();

  useEffect(() => {
    resetScan(); 
  }, []);

  useEffect(() => {
    if (user || localStorage.getItem('token')) {
        navigate('/scan');
    }
  }, [user, navigate]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const endpoint = isRegister ? 'register' : 'login';
      const payload = isRegister ? { email, password, username } : { email, password };
      
      const res = await api.post(`/auth/${endpoint}`, payload);
      login({ email, username: res.data.username, user_type: res.data.user_type }, res.data.access_token);
      toast.success(`Welcome back, ${res.data.username}!`);
      navigate('/scan'); 
    } catch (err) {
      toast.error("Error: " + (err.response?.data?.detail || "Invalid Credentials"));
    }
  };

  const handleGuest = async () => {
    try {
        const res = await api.post('/auth/guest-login');
        login({ email: "guest", username: "Guest User", user_type: "guest" }, res.data.access_token);
        toast.info("Logged in as Guest Mode");
        navigate('/scan'); 
    } catch (err) {
        toast.error("Guest login failed");
    }
  };

  return (
    <div className="min-h-screen pt-40 flex items-center justify-center relative overflow-hidden">
      <div className="bright-login-bg"></div>
      <div className="orb orb-1"></div>
      <div className="orb orb-2"></div>

      <div className="glass-panel p-8 rounded-2xl w-full max-w-md relative z-10 mx-4 shadow-[0_8px_32px_rgba(31,38,135,0.37)] border border-white/20">
        <h2 className="text-3xl font-bold text-center mb-8 text-white drop-shadow-md">
          {isRegister ? "Create Account" : "Access Portal"}
        </h2>

        <form onSubmit={handleSubmit} className="space-y-6">
          {isRegister && (
             <input type="text" placeholder="Username" className="w-full bg-white/10 border border-white/20 p-3 rounded-lg text-white placeholder-gray-300 focus:border-white focus:outline-none focus:ring-2 focus:ring-white/20 transition backdrop-blur-sm" onChange={e => setUsername(e.target.value)} required />
          )}
          <input type="email" placeholder="Email" className="w-full bg-white/10 border border-white/20 p-3 rounded-lg text-white placeholder-gray-300 focus:border-white focus:outline-none focus:ring-2 focus:ring-white/20 transition backdrop-blur-sm" onChange={e => setEmail(e.target.value)} required />
          <input type="password" placeholder="Password" className="w-full bg-white/10 border border-white/20 p-3 rounded-lg text-white placeholder-gray-300 focus:border-white focus:outline-none focus:ring-2 focus:ring-white/20 transition backdrop-blur-sm" onChange={e => setPassword(e.target.value)} required />

          <button type="submit" className="w-full bg-white text-blue-900 font-black py-3 rounded-lg hover:bg-gray-100 transition transform hover:scale-[1.02] shadow-lg">
            {isRegister ? "Sign Up" : "Secure Login"}
          </button>
        </form>

        <div className="mt-6 flex items-center justify-between text-sm text-gray-200">
          <button onClick={() => setIsRegister(!isRegister)} className="hover:text-white underline font-medium transition">
            {isRegister ? "Already have an account?" : "Need an account?"}
          </button>
        </div>

        <div className="mt-8 border-t border-white/20 pt-6">
           <button onClick={handleGuest} className="w-full border border-white/40 py-3 rounded-lg text-white hover:bg-white/20 transition font-medium backdrop-blur-md">
             Continue as Guest
           </button>
           <p className="text-xs text-center mt-2 text-gray-200 opacity-80">Guest mode limits history and downloads.</p>
        </div>
      </div>
    </div>
  );
};

export default Login;