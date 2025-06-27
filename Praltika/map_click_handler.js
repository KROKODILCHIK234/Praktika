// JavaScript код для обработки кликов на 3D карте
document.addEventListener('DOMContentLoaded', function() {
    // Функция для отправки координат клика в Python через Streamlit
    function handleMapClick(event) {
        // Получаем координаты клика
        const clickEvent = event.points[0];
        
        // Проверяем, есть ли координаты
        if (clickEvent) {
            // Создаем всплывающее окно с координатами
            const lat = clickEvent.lat || clickEvent.y;
            const lon = clickEvent.lon || clickEvent.x;
            
            if (lat && lon) {
                // Создаем всплывающее окно с координатами
                const alertDiv = document.createElement('div');
                alertDiv.style.position = 'fixed';
                alertDiv.style.top = '10px';
                alertDiv.style.left = '50%';
                alertDiv.style.transform = 'translateX(-50%)';
                alertDiv.style.backgroundColor = '#FFC107';
                alertDiv.style.color = '#212121';
                alertDiv.style.padding = '15px';
                alertDiv.style.borderRadius = '5px';
                alertDiv.style.zIndex = '1000';
                alertDiv.style.boxShadow = '0 2px 10px rgba(0,0,0,0.2)';
                alertDiv.style.maxWidth = '90%';
                alertDiv.style.textAlign = 'center';
                
                alertDiv.innerHTML = `
                    <h3 style="margin: 0 0 10px 0;">📍 Координаты клика</h3>
                    <p style="margin: 5px 0;">Широта: <b>${lat.toFixed(4)}</b></p>
                    <p style="margin: 5px 0;">Долгота: <b>${lon.toFixed(4)}</b></p>
                `;
                
                // Добавляем кнопку закрытия
                const closeButton = document.createElement('button');
                closeButton.textContent = '✖';
                closeButton.style.position = 'absolute';
                closeButton.style.top = '5px';
                closeButton.style.right = '5px';
                closeButton.style.border = 'none';
                closeButton.style.background = 'none';
                closeButton.style.fontSize = '16px';
                closeButton.style.cursor = 'pointer';
                closeButton.style.color = '#212121';
                
                closeButton.onclick = function() {
                    document.body.removeChild(alertDiv);
                };
                
                alertDiv.appendChild(closeButton);
                
                // Добавляем в DOM
                document.body.appendChild(alertDiv);
                
                // Автоматически удаляем через 5 секунд
                setTimeout(() => {
                    if (document.body.contains(alertDiv)) {
                        document.body.removeChild(alertDiv);
                    }
                }, 5000);
            }
        }
    }
    
    // Функция для поиска карты Plotly и добавления обработчика событий
    function setupMapClickHandler() {
        // Ищем все элементы Plotly
        const plotlyElements = document.querySelectorAll('.js-plotly-plot');
        
        if (plotlyElements.length > 0) {
            for (const element of plotlyElements) {
                // Проверяем, что это карта (имеет geo слой)
                if (element.querySelector('.geo')) {
                    // Добавляем обработчик события клика
                    element.on('plotly_click', handleMapClick);
                }
            }
        } else {
            // Если элементы не найдены, пробуем позже
            setTimeout(setupMapClickHandler, 1000);
        }
    }
    
    // Запускаем настройку обработчика
    setupMapClickHandler();
}); 