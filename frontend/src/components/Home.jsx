import React, { useRef } from "react";
import transition from "../transition";
import { Link } from "react-router-dom";

const Home = () => {
  const howRef = useRef(null);
  const startRef = useRef(null);
  return (
    <div className="w-full h-screen">
      <div className="bga-amber-500 p-3.5 flex gap-1.5 h-[80%] mt-5">
        <div className="bga-red-600 w-1/2 flex flex-col justify-around">
          <div className="bga-amber-950 text-[7.5rem] leading-[7.5rem]">
            <div className="">Find Your</div>
            <div className="ml-[14rem]">Perfect</div>
            <div className="ml-[36rem]">Fit</div>
          </div>
          <div className="bga-blue-400 flex gap-7 mt-7 justify-center items-center">
            <Link to="/how">
              <div
                onMouseEnter={() => {
                  howRef.current.style.height = "100%";
                }}
                onMouseLeave={() => {
                  howRef.current.style.height = "0%";
                }}
                className="bg-black relative h-10 lg:w-60 md:w-60 md:h-12 cursor-pointer w-40"
              >
                <div
                  ref={howRef}
                  className="bg-[#FF9B05] transition-all ease-in absolute top-0 w-full"
                ></div>
                <div className="group h-full relative flex flex-col gap-2 justify-center px-5 text-4xl text-white text-center hover:text-black">
                  HOW?
                </div>
              </div>
            </Link>
            <Link to="/upload">
              <div
                onMouseEnter={() => {
                  startRef.current.style.height = "100%";
                }}
                onMouseLeave={() => {
                  startRef.current.style.height = "0%";
                }}
                className="bg-black relative h-10 lg:w-[32rem] md:w-60 md:h-12 cursor-pointer w-40"
              >
                <div
                  ref={startRef}
                  className="bg-[#FF9B05] transition-all ease-in absolute top-0 w-full"
                ></div>
                <div className="group h-full relative flex gap-6 justify-center items-center px-5 py-2 text-4xl text-white text-center hover:text-black">
                  Upload & Check Your Fit
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="32"
                    height="30"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="lucide lucide-move-right-icon lucide-move-right text-white group-hover:text-black transition-colors"
                  >
                    <path d="M18 8L22 12L18 16" />
                    <path d="M2 12H22" />
                  </svg>
                </div>
              </div>
            </Link>
          </div>
        </div>
        <div className="bga-green-500 w-[40%] flex justify-center items-center">
          <img
            src="https://illustrations.popsy.co/amber/home-office.svg"
            alt=""
          />
        </div>
      </div>
    </div>
  );
};

export default transition(Home);
