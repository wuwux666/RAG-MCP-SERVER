import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/chat' },
    { path: '/chat', name: 'chat', component: () => import('../views/ChatView.vue') },
    { path: '/documents', name: 'documents', component: () => import('../views/DocumentsView.vue') },
    { path: '/traces', name: 'traces', component: () => import('../views/TracesView.vue') },
    { path: '/evaluation', name: 'evaluation', component: () => import('../views/EvaluationView.vue') },
  ]
})

export default router
