from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/')
def home():
    return '<h1>Welcome to My Simple Flask App</h1><p>Go to <a href="/form">Form Page</a></p>'

@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        name = request.form.get('name')
        return f'<h2>Hello, {name}!</h2><p><a href="/">Back to Home</a></p>'
    return '''
        <h1>Form Page</h1>
        <form method="post">
            <label for="name">Enter your name:</label>
            <input type="text" id="name" name="name" required>
            <input type="submit" value="Submit">
        </form>
        <p><a href="/">Back to Home</a></p>
    '''

if __name__ == '__main__':
    app.run(debug=True)
