import { NextRequest, NextResponse } from 'next/server'
import { headers } from 'next/headers'

export async function POST(request: NextRequest) {
  try {
    // Note: Vercel handles request size limits at the platform level
    // If the request is too large, it will be rejected before reaching this code

    const formData = await request.formData()
    const files = formData.getAll('files') as File[]
    const goal = formData.get('goal') as string

    if (!files || files.length === 0) {
      return NextResponse.json(
        { error: 'No files uploaded' },
        { status: 400 }
      )
    }

    if (!goal) {
      return NextResponse.json(
        { error: 'Research goal is required' },
        { status: 400 }
      )
    }

    // Validate individual file sizes
    const MAX_FILE_SIZE = 10 * 1024 * 1024 // 10MB per file
    const MAX_TOTAL_SIZE = 20 * 1024 * 1024 // 20MB total
    
    let totalSize = 0
    for (const file of files) {
      if (file.size > MAX_FILE_SIZE) {
        return NextResponse.json(
          { 
            error: 'File too large',
            details: `${file.name} exceeds 10MB limit. Current size: ${(file.size / 1024 / 1024).toFixed(2)}MB`
          },
          { status: 413 }
        )
      }
      totalSize += file.size
    }
    
    if (totalSize > MAX_TOTAL_SIZE) {
      return NextResponse.json(
        { 
          error: 'Total size too large',
          details: `Combined files exceed 20MB limit. Current size: ${(totalSize / 1024 / 1024).toFixed(2)}MB`
        },
        { status: 413 }
      )
    }

    // Connect to FastAPI backend for real PDF processing
    const backendUrl = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL || (
      process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}/api/backend` : 'http://localhost:7861'
    )
    
    try {
      // Create form data to send to FastAPI backend
      const backendFormData = new FormData()
      
      // Add files to form data
      for (const file of files) {
        backendFormData.append('files', file)
      }
      
      // Add research goal
      backendFormData.append('goal', goal)
      
      console.log(`Calling FastAPI backend at ${backendUrl}/summarize-pdfs`)
      console.log(`Processing ${files.length} files with goal: "${goal}"`)
      
      // Call FastAPI backend (OpenAI-powered)
      const backendResponse = await fetch(`${backendUrl}/summarize-pdfs`, {
        method: 'POST',
        body: backendFormData,
        // Don't set Content-Type header - let fetch set it for multipart/form-data
      })
      
      if (!backendResponse.ok) {
        const errorText = await backendResponse.text()
        console.error('Backend response error:', backendResponse.status, errorText)
        throw new Error(`Backend processing failed: ${backendResponse.status} ${backendResponse.statusText}`)
      }
      
      const backendData = await backendResponse.json()
      console.log('Backend processing successful:', backendData)
      
      // Transform backend response to match frontend expectations
      const summaries = backendData.summaries?.map((summary: any) => ({
        filename: summary.filename || summary.file_name || 'Unknown',
        title: summary.title || `Analysis of ${summary.filename || summary.file_name}`,
        summary: summary.summary || summary.content || 'No summary available'
      })) || []
      
      return NextResponse.json({
        goal: backendData.goal || goal,
        summaries,
        status: 'success',
        message: 'PDF processing completed successfully',
        backend_response: backendData // Include full backend response for debugging
      })
      
    } catch (backendError) {
      console.error('Error calling FastAPI backend:', backendError)
      
      // Return error with details about backend connectivity
      return NextResponse.json({
        error: 'Backend processing failed',
        details: backendError instanceof Error ? backendError.message : 'Unknown backend error',
        backend_url: backendUrl,
        suggestion: 'Make sure the FastAPI backend is running on port 7860'
      }, { status: 502 })
    }

  } catch (error) {
    console.error('Error processing PDFs:', error)
    return NextResponse.json(
      { 
        error: 'Failed to process PDFs',
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    )
  }
}

export async function GET() {
  return NextResponse.json({
    message: 'PDF Summarization API',
    status: 'active',
    endpoints: {
      'POST /api/summarize-pdfs': 'Upload PDFs and get summaries'
    }
  })
}