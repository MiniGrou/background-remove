from fastapi import FastAPI, UploadFile, Response
from fastapi.responses import StreamingResponse
import uvicorn
from backgroundremover.bg import remove
app = FastAPI()


@app.post("/")
def index(file: UploadFile):
    return StreamingResponse(content=remove(file.file.read(), model_name='u2net'), media_type="image/png")


if __name__ == '__main__':
    uvicorn.run('server:app', port=9090, reload=True)