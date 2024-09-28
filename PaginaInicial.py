from flask import render_template_string, url_for

def render_pagina_inicial():
    html = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>API Documentation</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style>
            body {
                padding-top: 50px;
            }
            .endpoint {
                border-left: 5px solid #007BFF;
                padding-left: 15px;
                margin-bottom: 30px;
            }
            .method {
                font-weight: bold;
                color: #fff;
                background-color: #007BFF;
                padding: 5px 10px;
                border-radius: 4px;
                margin-right: 10px;
            }
            pre {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>

    <div class="container">
        <h1 class="text-center">API Documentation</h1>
        <p class="lead text-center">Endpoints da API.</p>

        <!-- Endpoint: Register -->
        <div class="endpoint">
            <h2>Register Image</h2>
            <p><span class="method">POST</span> <strong>/registrar</strong></p>
            <p>This endpoint allows you to register a new image with its feature vector.</p>

            <h5>Request Parameters:</h5>
            <ul>
                <li><strong>file</strong> (required, form-data): The image file to register.</li>
                <li><strong>nome</strong> (required, form-data): The name associated with the image.</li>
            </ul>

            <h5>Response Example:</h5>
            <pre>
{
    "message": "Registro salvo!",
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "nome": "Cachorro"
}
            </pre>
        </div>

        <!-- Endpoint: Identify -->
        <div class="endpoint">
            <h2>Identify Image</h2>
            <p><span class="method">POST</span> <strong>/identificar</strong></p>
            <p>This endpoint compares an image with the registered images and returns the closest match.</p>

            <h5>Request Parameters:</h5>
            <ul>
                <li><strong>file</strong> (required, form-data): The image file to identify.</li>
            </ul>

            <h5>Response Example:</h5>
            <pre>
{
    "message": "Registro mais próximo encontrado",
    "registro": {
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "nome": "Cachorro",
        "feature_vector": [...]
    }
}
            </pre>
        </div>

        <!-- Endpoint: Verify by ID -->
        <div class="endpoint">
            <h2>Verify Image by ID</h2>
            <p><span class="method">POST</span> <strong>/verificar?id=</strong></p>
            <p>This endpoint allows you to check if an image matches a specific record by ID.</p>

            <h5>Request Parameters:</h5>
            <ul>
                <li><strong>id</strong> (required, query parameter): The ID of the image to verify.</li>
                <li><strong>file</strong> (required, form-data): The image file to compare.</li>
            </ul>

            <h5>Response Example:</h5>
            <pre>
{
    "message": "Verificação concluída",
    "resultado": "Correspondência encontrada"
}
            </pre>
        </div>
    </div>

    </body>
    </html>
    '''
    return render_template_string(html)
