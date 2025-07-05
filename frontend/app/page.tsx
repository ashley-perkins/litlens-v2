import { LitLensUploader } from "@/components/lit-lens-uploader"

export default function Home() {
  return (
    <main className="min-h-screen bg-[#f8fafc] flex flex-col items-center justify-center p-4 md:p-8">
      <div className="w-full max-w-3xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-[#1F2B3A] mb-2">LitLens</h1>
          <p className="text-lg text-gray-600 mb-4">Summarize scientific PDFs with AI</p>
        </div>

        <LitLensUploader />

        <div className="mt-12 text-center text-sm text-gray-500">
          <p>Upload your papers, get instant summaries, and move forward with clarity.</p>
        </div>
      </div>
    </main>
  )
}
