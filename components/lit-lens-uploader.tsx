"use client"

import type React from "react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card } from "@/components/ui/card"
import { Loader2, Upload, FileText, Brain } from "lucide-react"
import { cn } from "@/lib/utils"

export function LitLensUploader() {
  const [files, setFiles] = useState<File[]>([])
  const [researchGoal, setResearchGoal] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [summaries, setSummaries] = useState<
    { filename: string; title?: string; summary: string }[]
  >([])
  const [isDragging, setIsDragging] = useState(false)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(Array.from(e.target.files))
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    if (e.dataTransfer.files) {
      const droppedFiles = Array.from(e.dataTransfer.files).filter(
        (f) => f.type === "application/pdf"
      )
      setFiles(droppedFiles)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (files.length === 0 || !researchGoal) return

    setIsLoading(true)
    setSummaries([])

    try {
      const formData = new FormData()
      files.forEach((file) => formData.append("files", file))
      formData.append("goal", researchGoal)

      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "https://ashley-perkins-litlens.hf.space"
      const res = await fetch(`${apiUrl}/summarize-pdfs`, {
        method: "POST",
        body: formData,
      })

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({ detail: "Unknown error occurred" }))
        throw new Error(errorData.detail || `HTTP ${res.status}: ${res.statusText}`)
      }

      const data = await res.json()
      if (Array.isArray(data.summaries)) {
        setSummaries(
          data.summaries.map((item: any) => ({
            filename: item.filename || "Untitled",
            title: item.title,
            summary: item.summary || "No summary returned.",
          }))
        )
      } else {
        setSummaries([])
      }
    } catch (err) {
      console.error("Error fetching summary:", err)
      const errorMessage = err instanceof Error ? err.message : "Something went wrong while summarizing."
      setSummaries([{ filename: "Error", summary: errorMessage }])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="w-full">
      <form onSubmit={handleSubmit} className="space-y-6">
        <div
          className={cn(
            "border-2 border-dashed rounded-lg p-6 transition-colors",
            isDragging ? "border-[#1F2B3A] bg-blue-50" : "border-gray-300",
            files.length > 0 ? "bg-blue-50" : "bg-white"
          )}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="flex flex-col items-center justify-center space-y-4">
            <div className="p-3 rounded-full bg-blue-100">
              {files.length > 0 ? <FileText className="h-8 w-8 text-[#1F2B3A]" /> : <Upload className="h-8 w-8 text-[#1F2B3A]" />}
            </div>

            <div className="text-center">
              <h3 className="text-lg font-medium text-gray-900">
                {files.length > 0 ? `${files.length} PDF${files.length > 1 ? "s" : ""} selected` : "Upload your PDFs"}
              </h3>
              <p className="text-sm text-gray-500 mt-1">
                {files.length > 0 ? "Files selected" : "Drag and drop or click to select PDF files"}
              </p>
            </div>

            <Button
              type="button"
              variant="outline"
              onClick={() => document.getElementById("file-upload")?.click()}
              className="mt-2"
            >
              {files.length > 0 ? "Replace Files" : "Select PDFs"}
            </Button>

            {files.length > 0 && (
              <Button type="button" variant="outline" onClick={() => setFiles([])} className="mt-2">
                Clear Files
              </Button>
            )}

            <input
              id="file-upload"
              type="file"
              accept="application/pdf"
              multiple
              onChange={handleFileChange}
              className="hidden"
            />
          </div>
        </div>

        <div className="space-y-2">
          <label htmlFor="research-goal" className="block text-sm font-medium text-gray-700">
            Research Goal
          </label>
          <Textarea
            id="research-goal"
            placeholder="e.g. Identify biomarkers for appendiceal neoplasms"
            value={researchGoal}
            onChange={(e) => setResearchGoal(e.target.value)}
            className="min-h-[100px] bg-white"
          />
        </div>

        <Button
          type="submit"
          className="w-full bg-[#1F2B3A] hover:bg-[#2d3b4d]"
          disabled={files.length === 0 || !researchGoal || isLoading}
        >
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Summarizing...
            </>
          ) : (
            <>
              <Brain className="mr-2 h-4 w-4" />
              Summarize
            </>
          )}
        </Button>
      </form>

      {(isLoading || summaries.length > 0) && (
        <div className="mt-8">
          <h3 className="text-lg font-medium text-gray-900 mb-3">Summary</h3>
          <Card className="p-6 bg-white border border-gray-200 rounded-lg space-y-6">
            {isLoading ? (
              <div className="flex flex-col items-center justify-center py-8">
                <Loader2 className="h-8 w-8 text-[#1F2B3A] animate-spin mb-4" />
                <p className="text-gray-500">Analyzing your document(s)...</p>
              </div>
            ) : (
              summaries.map((s, i) => (
                <div key={i}>
                  <h4 className="text-md font-semibold text-[#1F2B3A] mb-1">
                    {s.title || s.filename}
                  </h4>
                  {s.title && s.filename !== s.title && (
                    <p className="text-sm text-gray-500 mb-2">File: {s.filename}</p>
                  )}
                  <p className="whitespace-pre-line text-gray-700">{s.summary}</p>
                  {i < summaries.length - 1 && <hr className="my-4 border-gray-200" />}
                </div>
              ))
            )}
          </Card>
        </div>
      )}
    </div>
  )
}