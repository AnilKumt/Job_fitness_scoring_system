import React,{useEffect,useState} from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { CheckCircle2, XCircle, FileText } from "lucide-react";
import { usePrediction } from "../context/PredictionContext";
import transition from "../transition";

const Output = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { resumeFile, fit } = usePrediction();
  const [pdfUrl, setPdfUrl] = useState(null);

  useEffect(() => {
    if (!resumeFile) {
      navigate("/upload");
    } else {
      const blobUrl = URL.createObjectURL(resumeFile);
      setPdfUrl(blobUrl);

      return () => URL.revokeObjectURL(blobUrl); 
    }
  }, [resumeFile, navigate]);


  return (
    <div className="flex flex-col md:flex-row items-center justify-center w-full h-[90%] bg-white px-6 py-10 gap-8">
      
      <div className="flex flex-col items-center w-full md:w-1/2 bg-gray-50 border rounded-2xl p-6 shadow-sm">
        <h2 className="text-xl font-bold mb-4">Your Resume</h2>
        <div className="flex flex-col items-center gap-3 border p-4 rounded-lg shadow-sm bg-white w-full h-[500px]">
          {pdfUrl ? (
            <iframe
              src={pdfUrl}
              title="Resume Preview"
              className="w-full h-full rounded-lg"
            />
          ) : (
            <div className="flex flex-col items-center text-gray-500">
              <FileText size={40} />
              <span className="text-sm">No resume uploaded</span>
            </div>
          )}
        </div>
      </div>

      
      <div className="flex flex-col items-center w-full md:w-1/2 bg-gray-50 border rounded-2xl p-6 shadow-sm">
        {fit ? (
          <div className="flex flex-col items-center animate-fadeIn">
            <CheckCircle2 size={60} className="text-green-500 mb-2" />
            <h2 className="text-2xl font-bold text-green-600">🎉 Congratulations!</h2>
            <p className="text-gray-700 mt-2 text-center">
              You are a great fit for this role. Go ahead and apply with confidence!
            </p>
            <img
              src="https://illustrations.popsy.co/amber/work-party.svg"
              alt="Congrats doodle"
              className="mt-4 w-52"
            />
          </div>
        ) : (
          <div className="flex flex-col items-center animate-fadeIn">
            <XCircle size={60} className="text-red-500 mb-2" />
            <h2 className="text-2xl font-bold text-red-600">Not a Perfect Fit Yet!</h2>
            <p className="text-gray-700 mt-2 text-center">
              Don’t worry, every rejection is redirection. Keep improving and the right opportunity will come. 🚀
            </p>
            <img
              src="https://illustrations.popsy.co/amber/studying.svg"
              alt="Motivation doodle"
              className="mt-4 w-52"
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default transition(Output);
