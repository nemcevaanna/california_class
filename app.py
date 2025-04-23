import streamlit as st  # Импортируем библиотеку Streamlit для создания веб-приложения
import requests  # Импортируем библиотеку requests для выполнения HTTP-запросов

# Устанавливаем заголовок приложения с иконкой дома
st.title('🏠 Прогноз цены на жильё в Калифорнии')

# Выводим описание приложения с инструкцией для пользователя
st.markdown("Введите параметры дома для получения прогноза цены:")

# Создаем две колонки для компактного размещения полей ввода
col1, col2 = st.columns(2)

# В первой колонке размещаем первые четыре поля ввода
with col1:
    MedInc = st.number_input(
        "Медианный доход (в десятках тысяч $):",
        min_value=0.0, max_value=15.0, value=5.0, step=0.1
    )  # Поле для ввода медианного дохода в районе (в десятках тысяч долларов)
    HouseAge = st.number_input(
        "Возраст дома (лет):",
        min_value=1, max_value=52, value=20, step=1
    )  # Поле для ввода возраста дома в годах
    AveRooms = st.number_input(
        "Среднее количество комнат:",
        min_value=1.0, max_value=15.0, value=5.0, step=1.0
    )  # Поле для ввода среднего количества комнат в доме
    AveBedrms = st.number_input(
        "Среднее количество спален:",
        min_value=1.0, max_value=5.0, value=1.0, step=1.0
    )  # Поле для ввода среднего количества спален в доме

# Во второй колонке размещаем оставшиеся поля ввода
with col2:
    Population = st.number_input(
        "Население района:",
        min_value=1, max_value=40000, value=1000, step=1
    )  # Поле для ввода численности населения района
    AveOccup = st.number_input(
        "Среднее количество жильцов:",
        min_value=1.0, max_value=1243.0, value=3.0, step=0.1
    )  # Поле для ввода среднего количества жильцов в доме
    Latitude = st.number_input(
        "Широта:",
        min_value=32.0, max_value=42.0, value=36.0, step=0.01
    )  # Поле для ввода географической широты района
    Longitude = st.number_input(
        "Долгота:",
        min_value=-124.0, max_value=-114.0, value=-120.0, step=0.01
    )  # Поле для ввода географической долготы района

# Проверяем, нажата ли кнопка "Получить прогноз"
if st.button("Получить прогноз"):
    # Создаем словарь с данными, введенными пользователем
    data = {
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude
    }
    # Определяем URL-адрес API для получения прогноза
    url = 'https://california-class.onrender.com/predict'
    # Отправляем POST-запрос к API с данными в формате JSON
    response = requests.post(url, json=data)

    # Проверяем статус ответа от сервера
    if response.status_code == 200:
        try:
            # Преобразуем ответ из формата JSON в словарь
            data = response.json()
            # Извлекаем значение прогноза цены из ответа
            prediction = data.get('prediction')
            if prediction is not None:
                # Если прогноз получен, выводим его пользователю с форматированием
                st.success(f'💰 **Прогнозируемая цена: {prediction * 1000:.2f}$**')
            else:
                # Если в ответе отсутствует поле 'prediction', выводим сообщение об ошибке
                st.error("Ошибка: Ответ API не содержит прогноз.")
        except ValueError:
            # Обрабатываем ошибку, если ответ не в формате JSON
            st.error("Ошибка: Неверный формат ответа от API.")
    else:
        # Если статус ответа не 200 OK, выводим соответствующее сообщение об ошибке
        st.error(f"Ошибка: API вернул статус {response.status_code}.")