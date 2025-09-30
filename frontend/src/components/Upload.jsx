import React, { useState,useRef } from "react";
import { useMutation } from "@tanstack/react-query";
import { Upload as UploadIcon, FileText, Trash2 } from "lucide-react";
import { useNavigate } from "react-router-dom";
import transition from "../transition";
import { usePrediction } from "../context/PredictionContext";
import { Link } from "react-router-dom";

const Upload = () => {
  const [resumeFile, setResumeFile] = useState(null);
  const [jobDescription, setJobDescription] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const checkFitRef = useRef(null);
  const { setResumeFile: setGlobalResume, setFit: setGlobalFit } =
    usePrediction();
  const navigate = useNavigate();

  const predictFit = async (formData) => {
    return new Promise((resolve) => {
      setTimeout(() => {
        const isFit = Math.random() > 0.5;
        resolve({ fit: isFit });
      }, 2000);
    });
  };

  const mutation = useMutation({
    mutationFn: predictFit,
    onSuccess: (data) => {
      setGlobalResume(resumeFile);
      setGlobalFit(data.fit);
      navigate("/output");
    },
  });

  const handleSubmit = () => {
  if (!resumeFile || !jobDescription.trim()) return;

  const formData = new FormData();
  formData.append("resume", resumeFile);
  formData.append("jobDescription", jobDescription);

  setIsLoading(true); 
  mutation.mutate(formData, {
    onSuccess: (data) => {
      setGlobalResume(resumeFile);
      setGlobalFit(data.fit);
      setIsLoading(false); 
      navigate("/output");
    },
  });
};

  const handleDeleteResume = () => {
    setResumeFile(null);
  };

  return (
    <div className="flex flex-col items-center justify-center w-full h-[90%] bg-white px-6 py-10 poppins-regular">
      <h1 className="text-5xl mb-16 text-center border-b-orange-500 border-b-2">
        Upload & Check Your Fit
      </h1>

      {isLoading ? (
        <div className="flex flex-col items-center justify-center min-h-[90%]">
          <h2 className="text-xl font-bold mb-4">Checking your fit...</h2>
          <img
            src="https://illustrations.popsy.co/amber/surreal-hourglass.svg"
            alt="Loading doodle"
            className="w-60 animate-bounce mt-36"
          />
        </div>
      ) : (
        <div className="w-full max-w-2xl bg-gray-50 border rounded-2xl p-6 shadow-sm">
          <div className="mb-4 relative">
            <label className="block text-xl font-medium mb-2">
              Upload Resume
            </label>
            {resumeFile ? (
              <div className="flex items-center justify-between border rounded-lg p-2">
                <FileText size={20} />
                <span className="ml-2">{resumeFile.name}</span>
                <Trash2
                  size={20}
                  className="cursor-pointer text-red-500"
                  onClick={handleDeleteResume}
                />
              </div>
            ) : (
              <input
                type="file"
                accept=".pdf,.doc,.docx"
                onChange={(e) => setResumeFile(e.target.files[0])}
                className="block w-full border rounded-lg p-2
             file:mr-4 file:py-2 file:px-4 file:text-sm
             file:bg-orange-500 file:text-black file:cursor-pointer file:border-black file:border-2"
              />
            )}
          </div>

          <div className="mb-4">
            <label className="block text-xl font-medium mb-2">
              Paste Job Description
            </label>
            <textarea
              rows="5"
              value={jobDescription}
              onChange={(e) => setJobDescription(e.target.value)}
              placeholder="Paste the job description here..."
              className="block w-full border rounded-lg p-2"
            ></textarea>
          </div>

          {/* <button
            onClick={handleSubmit}
            disabled={mutation.isLoading}
            className="w-full flex items-center justify-center gap-2 bg-black text-white rounded-full py-3 hover:bg-gray-900 transition"
          >
            {mutation.isLoading ? (
              "Checking..."
            ) : (
              <>
                <UploadIcon size={18} /> Check Fit
              </>
            )}
          </button> */}
          <Link to="/upload">
          <div
            onClick={() => {
              handleSubmit();
            }}
            onMouseEnter={() => {
              checkFitRef.current.style.height = "100%";
            }}
            onMouseLeave={() => {
              checkFitRef.current.style.height = "0%";
            }}
            className="bg-black relative h-10 lg:w-[38rem] md:w-60 md:h-12 cursor-pointer w-40"
          >
            <div
              ref={checkFitRef}
              className="bg-orange-500 transition-all ease-in absolute top-0 w-full"
            ></div>
            <div className="group h-full relative flex gap-7 justify-center items-center px-5 text-3xl text-white text-center hover:text-black rounded-4xl">
              <UploadIcon size={30} /> Check Fit
            </div>
          </div>
        </Link>
        </div>
      )}
    </div>
  );
};

export default transition(Upload);
