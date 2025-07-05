# File Size Handling Guide

## Problem Solved
Fixed 413 "Content Too Large" errors when uploading PDFs to LitLens by implementing comprehensive file size validation and user feedback.

## File Size Limits

### Current Limits:
- **Per File**: 10MB maximum
- **Total Upload**: 20MB maximum
- **Vercel Serverless**: ~4.5MB body limit (Pro accounts)

### Why These Limits:
1. **Vercel Serverless Functions**: Have built-in payload limits
2. **Performance**: Large files take longer to process
3. **User Experience**: Faster uploads and processing
4. **Cost Management**: Avoid excessive bandwidth usage

## Frontend Validation

### Pre-Upload Validation:
```typescript
// File size validation before upload
const validateFiles = (fileList: File[]) => {
  const errors: string[] = []
  
  // Check individual file sizes (10MB limit)
  for (const file of fileList) {
    const fileSizeMB = file.size / (1024 * 1024)
    if (fileSizeMB > MAX_FILE_SIZE_MB) {
      errors.push(`${file.name} is too large`)
    }
  }
  
  // Check total size (20MB limit)
  const totalSize = fileList.reduce((sum, file) => sum + file.size, 0)
  const totalSizeMB = totalSize / (1024 * 1024)
  if (totalSizeMB > MAX_TOTAL_SIZE_MB) {
    errors.push('Total file size too large')
  }
  
  return errors
}
```

### User Interface Features:
1. **File Size Display**: Shows individual and total file sizes
2. **Real-time Validation**: Validates on file selection/drop
3. **Clear Error Messages**: Specific feedback about size limits
4. **File Size Formatting**: Human-readable format (KB, MB, GB)

## Backend Validation

### API Route Protection:
```typescript
// Content-Length header check
const contentLength = request.headers.get('content-length')
if (contentLength && parseInt(contentLength) > 20 * 1024 * 1024) {
  return NextResponse.json({
    error: 'Request too large',
    details: 'Total file size exceeds 20MB limit'
  }, { status: 413 })
}

// Individual file validation
for (const file of files) {
  if (file.size > MAX_FILE_SIZE) {
    return NextResponse.json({
      error: 'File too large',
      details: `${file.name} exceeds 10MB limit`
    }, { status: 413 })
  }
}
```

## Error Handling

### 413 Error Responses:
- **Frontend**: Catches 413 status and shows user-friendly message
- **Backend**: Returns structured error with specific details
- **User Feedback**: Clear instructions on file size limits

### Error Message Examples:
- "Files too large for upload. Please ensure each file is under 10MB"
- "document.pdf is too large (12.2MB). Maximum size per file: 10MB"
- "Total file size too large (25.5MB). Maximum total: 20MB"

## User Experience Improvements

### File Selection UI:
```
Selected Files:
📄 document1.pdf    2.1 MB
📄 document2.pdf    1.8 MB
📄 document3.pdf    0.9 MB
─────────────────────────
Total Size:         4.8 MB
Limit: 10MB per file, 20MB total
```

### Validation States:
- ✅ **Valid**: Green indicators, upload enabled
- ⚠️ **Warning**: Yellow indicators, approaching limits
- ❌ **Invalid**: Red indicators, upload disabled

## Future Enhancements

### For Larger Files:
1. **Chunked Upload**: Split large files into smaller chunks
2. **Compression**: Client-side PDF compression
3. **Progressive Upload**: Upload files one at a time
4. **File Optimization**: Reduce PDF quality/resolution

### Advanced Features:
1. **Drag & Drop Progress**: Visual feedback during validation
2. **File Type Detection**: Validate actual PDF content
3. **Preview Generation**: Show PDF thumbnails
4. **Batch Processing**: Handle multiple files sequentially

## Testing Scenarios

### Test Cases:
1. ✅ Small files (< 1MB each)
2. ✅ Medium files (1-4MB each)
3. ❌ Large single file (> 4MB)
4. ❌ Multiple files exceeding total limit
5. ✅ Mixed sizes within limits
6. ❌ Non-PDF files
7. ✅ Edge case: exactly at limits

### Browser Testing:
- Chrome, Firefox, Safari, Edge
- Mobile browsers (iOS Safari, Android Chrome)
- Different screen sizes and resolutions

## Deployment Configuration

### Vercel Settings:
```json
{
  "functions": {
    "app/api/summarize-pdfs/route.ts": {
      "maxDuration": 60
    }
  }
}
```

### Environment Variables:
No additional environment variables needed for file size handling.

## Monitoring

### Metrics to Track:
- File upload success rate
- Average file sizes
- 413 error frequency
- User abandonment after size errors

### Logging:
- File size validation failures
- Successful uploads by size category
- Processing time by file size

## Best Practices

### For Users:
1. **Optimize PDFs**: Use PDF compression tools
2. **Split Large Documents**: Break into smaller sections
3. **Remove Unnecessary Content**: Images, annotations
4. **Use Text-based PDFs**: Avoid image-heavy documents

### For Developers:
1. **Validate Early**: Check sizes before upload starts
2. **Clear Messaging**: Specific error messages with limits
3. **Progressive Enhancement**: Works without JavaScript
4. **Graceful Degradation**: Fallback for unsupported features

This comprehensive file size handling ensures a smooth user experience while respecting platform limitations.