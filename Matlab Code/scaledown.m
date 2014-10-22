function result = scaledown(input)

result = (input-min(input)) ./(max(input)-min(input));

end
