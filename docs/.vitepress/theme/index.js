import DefaultTheme from 'vitepress/theme'
import 'katex/dist/katex.min.css'
import { defineAsyncComponent } from 'vue'
import './style.css'
import DpoCodeFocus from '../../chapter09_alignment/components/DpoCodeFocus.vue'
import Layout from './Layout.vue'
import NavCard from './components/NavCard.vue'
import NavGrid from './components/NavGrid.vue'
import PpoCodeFocus from '../../chapter07_ppo/components/PpoCodeFocus.vue'
import GrpoCodeFocus from '../../chapter09_grpo_rlvr/components/GrpoCodeFocus.vue'
import StepBar from './components/StepBar.vue'

const Mermaid = defineAsyncComponent(
  () => import('vitepress-plugin-mermaid/Mermaid.vue')
)

function loadFonts() {
  // Apple 风格：使用系统原生字体，不加载外部字体
  // SF Pro / PingFang SC / Hiragino Sans GB 等系统字体已足够精美
  if (typeof document === 'undefined') return
}

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp(ctx) {
    DefaultTheme.enhanceApp?.(ctx)
    ctx.app.component('DpoCodeFocus', DpoCodeFocus)
    ctx.app.component('GrpoCodeFocus', GrpoCodeFocus)
    ctx.app.component('NavCard', NavCard)
    ctx.app.component('NavGrid', NavGrid)
    ctx.app.component('PpoCodeFocus', PpoCodeFocus)
    ctx.app.component('StepBar', StepBar)
    ctx.app.component('Mermaid', Mermaid)
    loadFonts()
  }
}
