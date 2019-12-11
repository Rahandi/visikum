<!DOCTYPE html>
<html lang="{{ str_replace('_', '-', app()->getLocale()) }}">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">

        <title>Laravel</title>

        <!-- Fonts -->
        <link href="https://fonts.googleapis.com/css?family=Nunito:200,600" rel="stylesheet">

        <!-- Styles -->
        <style>
            html, body {
                background-color: #fff;
                color: #636b6f;
                font-family: 'Nunito', sans-serif;
                font-weight: 200;
                height: 100vh;
                margin: 0;
            }

            .full-height {
                height: 100vh;
            }

            .flex-center {
                align-items: center;
                display: flex;
                justify-content: center;
            }

            .position-ref {
                position: relative;
            }

            .top-right {
                position: absolute;
                right: 10px;
                top: 18px;
            }

            .content {
                text-align: center;
            }

            .title {
                font-size: 84px;
            }

            .links > a {
                color: #636b6f;
                padding: 0 25px;
                font-size: 13px;
                font-weight: 600;
                letter-spacing: .1rem;
                text-decoration: none;
                text-transform: uppercase;
            }

            .m-b-md {
                margin-bottom: 30px;
            }
        </style>
    </head>
    <body>
        <div class="flex-center position-ref full-height">
            @if (Route::has('login'))
                <div class="top-right links">
                    @auth
                        <a href="{{ url('/home') }}">Home</a>
                    @else
                        <a href="{{ route('login') }}">Login</a>

                        @if (Route::has('register'))
                            <a href="{{ route('register') }}">Register</a>
                        @endif
                    @endauth
                </div>
            @endif

            <div class="content">
                <div class="col-sm-8">
                    <video width="1280" height="720" autoplay loop muted controls>
                        <source src="http://10.151.33.18:8080/go.ogg" type="video/ogg">
                        Your browser does not support the video tag.
                    </video>
                </div>
                <div class="col-sm 4" id="history">
                    <table border="" style="text-align: center; width:100%">
                        <thead>
                            <th style="width: 25px;">Nama</th>
                            <th style="width: 25px;">Timestamp</th>
                        </thead>
                        <tbody id="tab_history">

                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        <script src="https://code.jquery.com/jquery-3.4.1.slim.min.js" integrity="sha384-J6qa4849blE2+poT4WnyKhv5vZF5SrPo0iEjwBvKU7imGFAV0wwj1yYfoRSJoZ+n" crossorigin="anonymous"></script>
        <script src="https://code.jquery.com/jquery-3.4.1.min.js" integrity="sha256-CSXorXvZcTkaix6Yvo6HppcZGetbYMGWSFlBw8HfCJo=" crossorigin="anonymous"></script>
        <script>
            let jsonData = new Array();
            const removeChildren = (node) => {
                while (node.firstChild) {
                    node.removeChild(node.firstChild);
                }
            }
            $(document).ready(function(){
                setInterval(function(){
                    $.ajax({
                        url : "{{route('ambil')}}",
                        type : "GET",
                        dataType : "json",
                        contentType: "application/json; charset=utf-8",
                        data: {
                            "id": 'id',
                            "nama": 'nama',
                            "timestamp": 'timestamp',
                            "_token": "{{ csrf_token() }}"
                        },
                        success: function(data){
                            jsonData = data;
                            console.log(jsonData);
                            let dataDiv = document.getElementById('tab_history');
                            let fragment = document.createDocumentFragment();
                            let i = 0;
                            for (const data of jsonData.reverse()) {
                                if (i <= 10){
                                    let rows = document.createElement('tr');
                                    let nama = document.createElement('td');
                                    nama.setAttribute('class', 'nama');
                                    nama.textContent = `${data.nama}`;
                                    let waktu = document.createElement('td');
                                    waktu.setAttribute('class', 'waktu');
                                    waktu.textContent = `${data.timestamp}`;
                                    fragment.appendChild(rows);
                                    rows.appendChild(nama);
                                    rows.appendChild(waktu);
                                    i++;
                                }
                                else{
                                    break;
                                }
                            }
                            removeChildren(dataDiv);
                            dataDiv.appendChild(fragment);
                        }
                    }, "json");
                })
            })
        </script>
    </body>
</html>
