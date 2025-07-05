/**
 * Simple test to verify API endpoints work correctly
 * Run with: node test-api.js
 */

const BASE_URL = process.env.BASE_URL || 'http://localhost:3000';

async function testHealthEndpoint() {
  try {
    console.log('🔍 Testing health endpoint...');
    const response = await fetch(`${BASE_URL}/api/health`);
    const data = await response.json();
    
    if (response.ok) {
      console.log('✅ Health endpoint working:', data);
      return true;
    } else {
      console.log('❌ Health endpoint failed:', response.status);
      return false;
    }
  } catch (error) {
    console.log('❌ Health endpoint error:', error.message);
    return false;
  }
}

async function testSummarizeEndpoint() {
  try {
    console.log('🔍 Testing summarize endpoint...');
    
    // Create a simple test FormData
    const formData = new FormData();
    const testFile = new File(['Test PDF content'], 'test.pdf', { type: 'application/pdf' });
    formData.append('files', testFile);
    formData.append('goal', 'Test research goal');
    
    const response = await fetch(`${BASE_URL}/api/summarize-pdfs`, {
      method: 'POST',
      body: formData
    });
    
    const data = await response.json();
    
    if (response.ok) {
      console.log('✅ Summarize endpoint working:', {
        status: data.status,
        summaryCount: data.summaries?.length || 0,
        goal: data.goal
      });
      return true;
    } else {
      console.log('❌ Summarize endpoint failed:', response.status, data);
      return false;
    }
  } catch (error) {
    console.log('❌ Summarize endpoint error:', error.message);
    return false;
  }
}

async function runTests() {
  console.log('🚀 Starting API tests...\n');
  
  const healthTest = await testHealthEndpoint();
  console.log('');
  
  const summarizeTest = await testSummarizeEndpoint();
  console.log('');
  
  if (healthTest && summarizeTest) {
    console.log('🎉 All tests passed! API is working correctly.');
  } else {
    console.log('⚠️  Some tests failed. Check the output above.');
  }
}

// Only run if this file is executed directly
if (typeof window === 'undefined' && require.main === module) {
  runTests();
}

module.exports = { testHealthEndpoint, testSummarizeEndpoint };