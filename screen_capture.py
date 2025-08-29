import cv2
import numpy as np
import pyautogui
import asyncio
import websockets
import base64

async def stream_screen(websocket):
    while True:
        screenshot = pyautogui.screenshot()
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = base64.b64encode(buffer).decode('utf-8')
        await websocket.send(frame_data)
        await asyncio.sleep(0.1)  

async def main():
    async with websockets.serve(stream_screen, "localhost", 8765):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())