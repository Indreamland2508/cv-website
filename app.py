from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# 1. Tải "bộ não" AI lên bộ nhớ của máy chủ
model = joblib.load('battery_ai_model.pkl')

# 2. Trang chủ: Hiển thị giao diện Web
@app.route('/')
def home():
    return render_template('index.html')

# 3. Cổng API: Nhận dữ liệu và Trả kết quả dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Nhận dữ liệu từ Web gửi lên
        data = request.get_json()
        
        # Đóng gói thành DataFrame
        input_df = pd.DataFrame({
            'cycle': [float(data['cycle'])],
            'ambient_temperature': [float(data['temp'])],
            'temp_gradient': [float(data['gradient'])]
        })
        
        # Gọi AI dự đoán
        soh = model.predict(input_df)[0]
        
        # Xử lý logic cảnh báo
        status = "AN TOÀN" if soh >= 80 else "CẢNH BÁO" if soh >= 70 else "NGUY HIỂM"
        
        # Trả kết quả về cho Web
        return jsonify({
            'success': True,
            'soh_value': round(soh, 2),
            'status': status
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Chạy server ở cổng 5000
    app.run(debug=True, port=5000)