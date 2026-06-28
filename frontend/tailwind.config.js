export default {
  content: ["./index.html", "./src/**/*.{vue,ts,js}"],
  theme: {
    extend: {
      colors: {
        sakura: {
          pink: '#FF91A4',
          light: '#FFB6C1',
          bg: '#FFF5F7',
          text: '#4A3040',
          muted: '#C9A0B0',
          accent: '#FF85A2',
        }
      },
      borderRadius: {
        'xl': '16px',
        '2xl': '20px',
      },
      boxShadow: {
        'sakura': '0 4px 15px rgba(255, 145, 164, 0.2)',
      },
      fontFamily: {
        sans: ['"Nunito"', '"PingFang SC"', '"Microsoft YaHei"', 'sans-serif'],
      }
    }
  }
}
