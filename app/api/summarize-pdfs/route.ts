import { NextRequest, NextResponse } from 'next/server'
import { headers } from 'next/headers'

export async function POST(request: NextRequest) {
  try {
    // Check content length before processing
    const contentLength = request.headers.get('content-length')
    console.log('=== CONTENT LENGTH DEBUG ===')
    console.log('Content-Length header:', contentLength)
    console.log('Content-Length in MB:', contentLength ? (parseInt(contentLength) / 1024 / 1024).toFixed(2) : 'N/A')
    console.log('Content-Length limit:', 25 * 1024 * 1024, 'bytes (25MB)')
    
    if (contentLength && parseInt(contentLength) > 25 * 1024 * 1024) { // 25MB limit to account for multipart overhead
      console.log('❌ Content-Length exceeds limit')
      return NextResponse.json(
        { 
          error: 'Request too large',
          details: 'Total file size exceeds 20MB limit. Please reduce file size or number of files.'
        },
        { status: 413 }
      )
    }
    console.log('✅ Content-Length is within limit')
    console.log('=== END CONTENT LENGTH DEBUG ===')

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
    
    console.log('=== API VALIDATION DEBUG ===')
    console.log('MAX_FILE_SIZE:', MAX_FILE_SIZE, 'bytes (', MAX_FILE_SIZE / 1024 / 1024, 'MB)')
    console.log('MAX_TOTAL_SIZE:', MAX_TOTAL_SIZE, 'bytes (', MAX_TOTAL_SIZE / 1024 / 1024, 'MB)')
    console.log('Number of files:', files.length)
    
    let totalSize = 0
    for (const file of files) {
      console.log(`API File: ${file.name}, Size: ${file.size} bytes, ${(file.size / 1024 / 1024).toFixed(2)}MB`)
      if (file.size > MAX_FILE_SIZE) {
        console.log(`❌ API: File ${file.name} exceeds limit`)
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
    
    console.log(`API Total size: ${totalSize} bytes, ${(totalSize / 1024 / 1024).toFixed(2)}MB`)
    if (totalSize > MAX_TOTAL_SIZE) {
      console.log(`❌ API: Total size exceeds limit`)
      return NextResponse.json(
        { 
          error: 'Total size too large',
          details: `Combined files exceed 20MB limit. Current size: ${(totalSize / 1024 / 1024).toFixed(2)}MB`
        },
        { status: 413 }
      )
    }
    
    console.log('✅ API: All files passed validation')
    console.log('=== END API DEBUG ===')

    // For now, return a simple response indicating the feature is working
    // In production, you would integrate with your Python backend or implement the logic here
    const summaries = files.map((file, index) => ({
      filename: file.name,
      title: `Analysis of ${file.name}`,
      summary: `This is a placeholder summary for ${file.name}. Research goal: ${goal}. 
      
The system has successfully received your PDF and research goal. In a production environment, this would:
1. Extract text from the PDF using advanced OCR/parsing
2. Apply AI-powered relevance filtering based on your research goal
3. Generate comprehensive summaries using language models
4. Provide detailed analysis and insights

File details:
- Name: ${file.name}
- Size: ${(file.size / 1024 / 1024).toFixed(2)} MB
- Type: ${file.type}

To enable full functionality, integrate with the Python backend modules or implement the summarization logic directly in this API route.`
    }))

    return NextResponse.json({
      goal,
      summaries,
      status: 'success',
      message: 'PDF processing completed successfully'
    })

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