import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8000/api',
  timeout: 60000,
})

export async function query(params: { query: string; top_k?: number; collection?: string }) {
  const { data } = await api.post('/query', params)
  return data
}

export async function agenticQuery(params: { query: string; max_rounds?: number; collection?: string }) {
  const { data } = await api.post('/agentic-query', params)
  return data
}

export async function listCollections() {
  const { data } = await api.get('/collections')
  return data
}

export async function uploadDocument(file: File, collection: string = 'default') {
  const form = new FormData()
  form.append('file', file)
  form.append('collection', collection)
  const { data } = await api.post('/documents/upload', form)
  return data
}

export async function listDocuments() {
  const { data } = await api.get('/documents')
  return data
}

export async function deleteDocument(docId: string) {
  const { data } = await api.delete(`/documents/${docId}`)
  return data
}

export async function listQueryTraces(limit: number = 50) {
  const { data } = await api.get('/traces/query', { params: { limit } })
  return data
}

export async function listIngestionTraces(limit: number = 20) {
  const { data } = await api.get('/traces/ingestion', { params: { limit } })
  return data
}
