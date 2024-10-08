
FROM node:18 as builder 

WORKDIR /app
COPY ./dist ./dist

FROM nginx 

COPY ./default.conf /etc/nginx/conf.d/default.conf 
COPY --from=builder /app/dist  /usr/share/nginx/html 
