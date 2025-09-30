import React, { useRef } from "react";
import transition from "../transition";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/all";
import { useGSAP } from "@gsap/react";
import { FileText, Settings, Brain, BarChart, CheckCircle } from "lucide-react";

gsap.registerPlugin(ScrollTrigger);
const How = () => {
  const wrapperRef = useRef(null);
  const containerRef = useRef(null);
  const maskRef = useRef(null);
  const sectionsRef = useRef([]);

  const addToRefs = (el) => {
    if (el && !sectionsRef.current.includes(el)) {
      sectionsRef.current.push(el);
    }
  };

  useGSAP(() => {
    const sections = sectionsRef.current;
    const mask = maskRef.current;
    const totalScroll = containerRef.current.scrollWidth - window.innerWidth;

    let scrollTween = gsap.to(sections, {
      xPercent: -108 * sections.length,
      ease: "none",
      scrollTrigger: {
        trigger: containerRef.current,
        pin: true,
        scrub: 1,
        end: `+=${totalScroll}`,
      },
    });

    gsap.to(mask, {
      attr: { width: "100%" },
      scrollTrigger: {
        trigger: wrapperRef.current,
        start: "top left",
        end: () => `+=${totalScroll}`,
        scrub: 1,
      },
    });

    sections.forEach((section) => {
      let texts = section.querySelectorAll(".anim");
      gsap.from(texts, {
        y: -130,
        opacity: 0,
        duration: 2,
        ease: "elastic",
        stagger: 0.1,
        scrollTrigger: {
          trigger: section,
          containerAnimation: scrollTween,
          start: "left center",
        },
      });
    });
  }, []);

  return (
    <>
      <div
        ref={wrapperRef}
        className="wrapper overflow-x-hidden relative  h-[80%] w-full poppins-regular -top-64"
      >
        <div
          ref={containerRef}
          className="container scrollx flex flex-row w-max  h-[80vh] mt-8"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 900 100"
            fill="none"
            className="absolute top-[12rem] left-[10vw] w-[50vw]"
          >
            <defs>
              <mask id="revealMask" className="w-0">
                <rect
                  id="maskRect"
                  ref={maskRef}
                  x="0"
                  y="0"
                  width="0"
                  height="100"
                  fill="white"
                />
              </mask>
            </defs>

            <line
              x1="0"
              y1="50"
              x2="800"
              y2="50"
              stroke="#ddd"
              strokeWidth="4"
            />
            {/* <circle cx="80" cy="50" r="8" fill="#ddd" /> */}
            {/* <circle cx="160" cy="50" r="8" fill="#ddd" />
            <circle cx="240" cy="50" r="8" fill="#ddd" />
            <circle cx="320" cy="50" r="8" fill="#ddd" />
            <circle cx="400" cy="50" r="8" fill="#ddd" />
            <circle cx="480" cy="50" r="8" fill="#ddd" /> */}
            {Array.from({ length: 6 }).map((_, i) => {
              const x = (800 / 5) * i; // 5 intervals for 6 circles
              return <circle key={i} cx={x} cy="50" r="8" fill="#ddd" />;
            })}

            {/* <g mask="url(#revealMask)">
              <line
                x1="0"
                y1="50"
                x2="800"
                y2="50"
                stroke="black"
                strokeWidth="4"
              />
              <circle cx="10" cy="50" r="8" fill="black" />
              <circle cx="160" cy="50" r="8" fill="black" />
              <circle cx="240" cy="50" r="8" fill="black" />
              <circle cx="320" cy="50" r="8" fill="black" />
              <circle cx="400" cy="50" r="8" fill="black" />
              <circle cx="480" cy="50" r="8" fill="black" />
              
            </g> */}
            <g mask="url(#revealMask)">
              <line
                x1="0"
                y1="50"
                x2="800"
                y2="50"
                stroke="black"
                strokeWidth="4"
              />
              {Array.from({ length: 6 }).map((_, i) => {
                const x = (800 / 5) * i; // 5 intervals for 6 circles
                return <circle key={i} cx={x} cy="50" r="8" fill="black" />;
              })}
            </g>
          </svg>

          <section
            ref={addToRefs}
            className="w-screen h-full poppins-regular flex-shrink-0 mt-72"
          >
            <span className="mx-[18rem] text-2xl uppercase tracking-wide border-b-3 border-b-orange-500">
              Step 1
            </span>
            <h1 className="mx-[18rem] text-[3em] font-bold flex items-center gap-3 mt-2">
              <FileText className="w-12 h-12 text-orange-500" /> Resume & Job
              Upload
            </h1>
            <p className="mx-[18rem] text-lg mt-6 max-w-3xl">
              Users upload their{" "}
              <span className="font-semibold text-orange-500">resume</span> and{" "}
              <span className="font-semibold text-orange-500">
                job description
              </span>
              . These files are securely sent to the backend for preprocessing.
            </p>
          </section>

          <section
            ref={addToRefs}
            className="w-screen h-full flex-shrink-0 mt-72 "
          >
            <div className="px-[10rem]">
              <span className="anim text-2xl uppercase tracking-wide border-b-3 border-b-orange-500">
                Step 2
              </span>
              <h1 className="anim text-[3em] font-bold flex items-center gap-3 mt-2">
                <Settings className="w-12 h-12 text-orange-500" /> Text
                Extraction
              </h1>
              <div className="anim grid grid-cols-1 md:grid-cols-2 gap-8 mt-10">
                <div className="p-6 rounded-2xl shadow-lg border-2 border-black">
                  <h2 className="text-xl font-semibold ">Tools Used</h2>
                  <p className="text-sm mt-2">
                    Libraries like{" "}
                    <span className="font-semibold text-orange-500">
                      PyPDF2
                    </span>
                    ,{" "}
                    <span className="font-semibold text-orange-500">docx</span>,
                    and{" "}
                    <span className="font-semibold text-orange-500">spaCy</span>{" "}
                    are used to parse resumes and job descriptions.
                  </p>
                </div>
              </div>
            </div>
          </section>

          <section
            ref={addToRefs}
            className="w-screen h-full flex-shrink-0 mt-72"
          >
            <div className="px-[10rem]">
              <span className="anim text-2xl uppercase tracking-wide border-b-3 border-b-orange-500">
                Step 3
              </span>
              <h1 className="anim text-[3em] font-bold flex items-center gap-3 mt-2">
                <Brain className="w-12 h-12 text-orange-500" /> NLP Feature
                Engineering
              </h1>
              <p className="anim text-lg mt-6 max-w-3xl">
                Extracted text is transformed into numerical representations
                like{" "}
                <span className="font-semibold text-orange-500">
                  TF-IDF vectors
                </span>{" "}
                and{" "}
                <span className="font-semibold text-orange-500">
                  word embeddings
                </span>{" "}
                so the model can understand semantic similarity.
              </p>
            </div>
          </section>

          <section
            ref={addToRefs}
            className="w-screen h-full flex-shrink-0 mt-72 mr-96"
          >
            <div className="anim px-[10rem] text-gray-900">
              <span className="anim text-2xl uppercase tracking-wide border-b-3 border-b-orange-500">
                Step 4
              </span>
              <h1 className="anim text-[3em] font-bold flex items-center gap-3 mt-2">
                <BarChart className="w-12 h-12 text-orange-500" /> Model
                Training
              </h1>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-10">
                <div className="bg-white p-6 rounded-2xl shadow-lg border-2 border-black">
                  <h2 className="anim text-xl font-semibold text-emerald-700">
                    Algorithms
                  </h2>
                  <p className="anim text-sm text-gray-700 mt-2">
                    We tested multiple models:{" "}
                    <span className="font-semibold text-orange-500">
                      Logistic Regression
                    </span>
                    ,{" "}
                    <span className="font-semibold text-orange-500">
                      Random Forest
                    </span>
                    , and{" "}
                    <span className="font-semibold text-orange-500">SVM</span>.
                    Final choice based on validation accuracy.
                  </p>
                </div>
                <div className="p-6 rounded-2xl shadow-lg border-2 border-black">
                  <h2 className="anim text-xl font-semibold text-emerald-700">
                    Workflow
                  </h2>
                  <p className="anim text-sm text-gray-700 mt-2">
                    Preprocessed features → Model training → Hyperparameter
                    tuning → Cross-validation.
                  </p>
                </div>
              </div>
            </div>
          </section>

          <section
            ref={addToRefs}
            className="w-screen h-full flex-shrink-0 mt-72 mr-96"
          >
            <div className="px-[10rem]">
              <span className="anim text-2xl uppercase tracking-wide border-b-3 border-b-orange-500">
                Step 5
              </span>
              <h1 className="anim text-[3em] font-bold flex items-center gap-3 mt-2">
                <BarChart className="w-12 h-12 text-orange-500" /> Evaluation
                Metrics
              </h1>
              <div className="anim grid grid-cols-2 md:grid-cols-4 gap-8 mt-10">
                {[
                  { label: "Accuracy", value: "92%" },
                  { label: "Precision", value: "0.91" },
                  { label: "Recall", value: "0.90" },
                  { label: "F1 Score", value: "0.905" },
                ].map((metric, idx) => (
                  <div
                    key={idx}
                    className="bg-white text-center p-6 rounded-2xl shadow-lg border-2 border-black"
                  >
                    <h2 className="text-3xl font-bold text-orange-500 anim">
                      {metric.value}
                    </h2>
                    <p className="text-gray-700 anim">{metric.label}</p>
                  </div>
                ))}
              </div>
            </div>
          </section>

          <section
            ref={addToRefs}
            className="w-screen h-full flex-shrink-0 mt-72 mr-96"
          >
            <div className="px-[10rem]">
              <span className="anim text-2xl uppercase tracking-wide border-b-3 border-b-orange-500">
                Step 6
              </span>
              <h1 className="anim text-[3em] font-bold flex items-center gap-3 mt-2">
                <CheckCircle className="w-12 h-12 text-orange-500" /> Final
                Prediction
              </h1>
              <p className="anim text-lg mt-6 max-w-3xl">
                The model finally predicts whether the candidate is a{" "}
                <span className="font-semibold text-emerald-700">Good Fit</span>{" "}
                or{" "}
                <span className="font-semibold text-red-500">
                  Not a Good Fit
                </span>{" "}
                for the uploaded job description.
              </p>
            </div>
          </section>
        </div>
      </div>
    </>
  );
};

export default transition(How);
