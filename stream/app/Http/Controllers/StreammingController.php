<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

use App\log;

class StreammingController extends Controller
{
    public function ajax()
    {
        $data = log::all();
        return $data;
    }
}
