from flask import Flask, request, render_template
from weather.sunlight_model import generate
import openai

openai.api_key = "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"


app = Flask(__name__)

x = generate()


def gen_prompt(devices, no_adults, no_children, month, house_age, house_size):
    insert_var = "tell me the average monthly power consumption of a household with the following devices: {}. Household with {} adults and {} children in Poland, in {}. Size of house: {}, age: {} years. Provide just a value, nothing else. Response: <power consumption>".format(
        devices, no_adults, no_children, month, house_size, house_age)
    return insert_var


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/calculate', methods=['POST'])
def calculate():
    devices = request.form['devices']
    adults = request.form['no_adults']
    children = request.form['no_children']
    month = request.form['month']
    house_age = request.form['house_age']
    house_size = request.form['house_size']
    square_m_solars = request.form['solar_area']

    prompt = gen_prompt(devices, adults, children,
                        month, house_age, house_size)

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt
    )

    res = response['choices'][0]['text']

    return render_template('result.html', result=res, sunlight=x)


if __name__ == '__main__':
    app.run()
